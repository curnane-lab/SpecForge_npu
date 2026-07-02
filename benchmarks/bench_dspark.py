#!/usr/bin/env python3
"""Run HF-only DSpark speculative-decoding evaluation.

This script evaluates a SpecForge-trained DSpark draft checkpoint directly, without
converting it to DeepSpec format or launching an SGLang server.

It can be run in single-process mode or under ``torchrun`` for multi-device
distributed evaluation. When distributed, each rank processes a shard of the
dataset and metrics are all-reduced before reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from specforge.data.template import TEMPLATE_REGISTRY
from specforge.modeling.draft.dflash import extract_context_feature
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
    sample_tokens,
)
from specforge.utils import get_local_device


@dataclass
class SampleMetrics:
    prompt: str
    output: str
    num_input_tokens: int
    num_output_tokens: int
    verify_count: int
    proposal_lengths: list[int]
    accepted_draft_lengths: list[int]
    acceptance_lengths: list[int]
    total_time_sec: float
    target_time_sec: float
    draft_time_sec: float


@dataclass
class DatasetMetrics:
    dataset: str
    num_samples: int
    draft_tokens_per_proposal: float
    acceptance_length: float
    verify_rate: float
    accept_rates_by_position: list[float | None]
    accuracy: float | None = None
    total_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    target_time_sec: float = 0.0
    draft_time_sec: float = 0.0


@dataclass
class _DatasetAggregate:
    """Raw counters used for distributed metric aggregation."""

    sample_count: int
    proposal_count: int
    acceptance_length_sum: int
    proposal_length_sum: int
    proposals_at_pos: list[int]
    accepted_at_pos: list[int]
    output_token_sum: int
    verify_count: int
    total_time_sec: float
    target_time_sec: float
    draft_time_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DSpark draft checkpoint")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--benchmark-list", nargs="*", default=[])
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--eval-data-limit", type=int, default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate per dataset. Applies to all "
        "dataset sources after --eval-data-limit is applied.",
    )
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--is-preformatted", action="store_true")
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument(
        "--distributed-backend",
        default=None,
        choices=["nccl", "gloo", "hccl"],
        help="Process-group backend for distributed evaluation. If omitted, the "
        "backend is inferred from the device type (nccl for cuda, hccl for npu, "
        "gloo otherwise).",
    )
    return parser.parse_args()


def _infer_backend(device_type: str) -> str:
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    return "gloo"


def init_distributed(
    args_device: str, distributed_backend: str | None
) -> tuple[torch.device, int, int]:
    """Initialize torch.distributed when launched with torchrun.

    Returns the local device, global rank and world size. In non-distributed
    mode returns (device, 0, 1).
    """
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        return resolve_device(args_device), 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # In distributed mode bind each process to its local rank device regardless
    # of the --device argument, which is what torchrun expects.
    if args_device != "auto" and not args_device.startswith(("cuda:", "npu:")):
        device_type = torch.device(args_device).type
    else:
        device_type = resolve_device("auto").type

    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_type == "npu":
        if not hasattr(torch, "npu"):
            raise RuntimeError(
                "NPU device requested but torch.npu is not available. "
                "Install the Ascend PyTorch adapter to run on NPU."
            )
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    else:
        device = torch.device("cpu")

    backend = distributed_backend or _infer_backend(device_type)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return device, rank, world_size


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return get_local_device()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _crop_cache(cache: Any, length: int) -> None:
    """Trim a key-value cache to ``length`` if it exposes ``crop``.

    Some target models (e.g. Qwen3.5) return a custom cache class that does not
    implement ``crop``. In that case we leave the cache untouched; the model
    still behaves correctly because the cache is keyed by sequence length.
    """
    if cache is not None and hasattr(cache, "crop"):
        cache.crop(length)


def trim_stop_tokens(
    output_ids: torch.Tensor,
    num_input_tokens: int,
    stop_token_ids: list[int] | None,
) -> torch.Tensor:
    if not stop_token_ids:
        return output_ids
    stop = torch.tensor(stop_token_ids, device=output_ids.device, dtype=output_ids.dtype)
    hits = torch.isin(output_ids[0, num_input_tokens:], stop).nonzero(as_tuple=True)[0]
    if hits.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + int(hits[0].item()) + 1]


def has_stop_token(token_ids: torch.Tensor, stop_token_ids: list[int] | None) -> bool:
    if not stop_token_ids:
        return False
    stop = torch.tensor(stop_token_ids, device=token_ids.device, dtype=token_ids.dtype)
    return bool(torch.isin(token_ids, stop).any().item())


def resolve_stop_token_ids(target_model: Any, tokenizer: Any) -> list[int] | None:
    eos_token_id = getattr(getattr(target_model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    result = []
    for token_id in eos_token_id:
        token_id = int(token_id)
        if token_id not in result:
            result.append(token_id)
    return result


def assert_no_final_target_layer(target_model: Any, target_layer_ids: list[int]) -> None:
    """Guard against using the final target decoder layer as a draft anchor.

    ``output_hidden_states`` stores the final *normalized* hidden state at the
    last layer, while target cache generation stores raw decoder-layer outputs.
    Using the last layer as a target anchor causes a train/eval mismatch, so we
    reject it early.
    """
    target_config = target_model.config
    if hasattr(target_config, "text_config"):
        target_config = target_config.text_config
    last_layer_id = int(target_config.num_hidden_layers) - 1
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    assert last_layer_id not in target_layer_ids, (
        "target_layer_ids must not include the final target decoder layer "
        f"{last_layer_id}. Use an earlier layer and regenerate the target cache "
        "and draft checkpoint."
    )


class ConfidenceRecorder:
    """Lightweight confidence-head calibration recorder.

    When ``confidence_threshold == 0.0`` and the draft model has a confidence
    head, this recorder collects per-position predicted prefix-acceptance
    probabilities and observed acceptance outcomes. After the dataset finishes
    it reports ECE-style calibration diagnostics.
    """

    def __init__(self, block_size: int, device: torch.device, num_bins: int = 20) -> None:
        self.block_size = int(block_size)
        self.num_bins = int(num_bins)
        # NPU kernels do not support float64; use float32 on NPU to avoid
        # implicit dtype casts that break scatter_add_.
        self.dtype = torch.float32 if device.type == "npu" else torch.float64
        self.counts = torch.zeros(
            (self.block_size, self.num_bins), dtype=self.dtype, device=device
        )
        self.pred_sums = torch.zeros_like(self.counts)
        self.target_sums = torch.zeros_like(self.counts)

    def observe(
        self,
        *,
        confidence_logits: torch.Tensor,
        accept_prefix_mask: torch.Tensor,
        effective_length: int,
    ) -> None:
        if effective_length <= 0:
            return
        step_probs = torch.sigmoid(confidence_logits[:, :effective_length]).squeeze(0)
        cumprod_pred = step_probs.cumprod(dim=0).to(self.dtype)
        prefix_label = (
            accept_prefix_mask[:, :effective_length].squeeze(0).to(self.dtype)
        )
        ones = torch.ones_like(cumprod_pred, dtype=self.dtype)

        bin_idx = (cumprod_pred * self.num_bins).long().clamp_(0, self.num_bins - 1)
        pos_idx = torch.arange(effective_length, device=cumprod_pred.device)
        flat = pos_idx * self.num_bins + bin_idx

        self.counts.view(-1).scatter_add_(0, flat, ones)
        self.pred_sums.view(-1).scatter_add_(0, flat, cumprod_pred)
        self.target_sums.view(-1).scatter_add_(0, flat, prefix_label)

    def reset(self) -> None:
        self.counts.zero_()
        self.pred_sums.zero_()
        self.target_sums.zero_()

    def all_reduce(self) -> None:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        for tensor in (self.counts, self.pred_sums, self.target_sums):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def compute(self) -> list[dict[str, Any]]:
        results = []
        for pos in range(self.block_size):
            counts = self.counts[pos]
            total = float(counts.sum().item())
            if total <= 1e-12:
                results.append(
                    {
                        "position": pos,
                        "total": 0.0,
                        "ece": float("nan"),
                        "mean_pred": float("nan"),
                        "mean_target": float("nan"),
                    }
                )
                continue
            denom = counts.clamp_min(1e-12)
            avg_pred = self.pred_sums[pos] / denom
            avg_target = self.target_sums[pos] / denom
            bin_err = (avg_pred - avg_target).abs()
            ece = float((bin_err * counts).sum().item() / total)
            results.append(
                {
                    "position": pos,
                    "total": total,
                    "ece": ece,
                    "mean_pred": float(self.pred_sums[pos].sum().item()) / total,
                    "mean_target": float(self.target_sums[pos].sum().item()) / total,
                }
            )
        return results


class DSparkHFGenerator:
    def __init__(
        self,
        *,
        target_model: Any,
        draft_model: DSparkDraftModel,
        tokenizer: Any,
        temperature: float,
        confidence_threshold: float,
        max_new_tokens: int,
        confidence_recorder: ConfidenceRecorder | None = None,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.max_new_tokens = int(max_new_tokens)
        self.embed_tokens = target_model.get_input_embeddings()
        self.lm_head = target_model.get_output_embeddings()
        if self.lm_head is None:
            self.lm_head = target_model.lm_head
        self.stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)
        self.confidence_recorder = confidence_recorder

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, prompt: str) -> SampleMetrics:
        target = self.target_model
        draft = self.draft_model
        input_ids = input_ids.to(target.device)
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + self.max_new_tokens
        block_size = int(draft.block_size)
        mask_token_id = int(draft.mask_token_id)

        total_start = time.perf_counter()
        target_time = 0.0
        draft_time = 0.0

        output_ids = torch.full(
            (1, max_length + block_size + 1),
            mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        past_key_values_draft = DynamicCache()

        t0 = time.perf_counter()
        initial = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        target_time += time.perf_counter() - t0
        past_key_values_target = initial.past_key_values
        output_ids[:, :num_input_tokens] = input_ids
        first_token = sample_tokens(initial.logits, self.temperature)
        output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

        acceptance_lengths: list[int] = []
        proposal_lengths: list[int] = []
        accepted_draft_lengths: list[int] = []
        if has_stop_token(first_token, self.stop_token_ids):
            output_ids = trim_stop_tokens(
                output_ids[:, : num_input_tokens + 1], num_input_tokens, self.stop_token_ids
            )
            return self._build_metrics(
                prompt,
                output_ids,
                num_input_tokens,
                [],
                [],
                [],
                total_time=time.perf_counter() - total_start,
                target_time=target_time,
                draft_time=draft_time,
            )

        target_hidden = extract_context_feature(initial.hidden_states, draft.target_layer_ids)
        start = num_input_tokens
        while start < max_length:
            proposal = self._propose(
                output_ids=output_ids,
                position_ids=position_ids,
                past_key_values_draft=past_key_values_draft,
                target_hidden=target_hidden,
                start=start,
                block_size=block_size,
                mask_token_id=mask_token_id,
            )
            draft_time += proposal["draft_time_sec"]

            t0 = time.perf_counter()
            verification = self._verify(
                verify_input_ids=proposal["verify_input_ids"],
                draft_probs=proposal["draft_probs"],
                position_ids=position_ids,
                past_key_values_target=past_key_values_target,
                start=start,
                draft_token_count=proposal["draft_token_count"],
                block_size=block_size,
                confidence_logits=proposal["confidence_logits"],
            )
            target_time += time.perf_counter() - t0
            past_key_values_target = verification["past_key_values_target"]

            accepted = verification["accepted_draft_tokens"]
            next_token = verification["next_token"]
            terminated_by_stop = verification["terminated_by_stop"]
            target_output = verification["target_output"]
            effective_proposal_length = verification["effective_proposal_length"]
            accept_prefix_mask = verification["accept_prefix_mask"]

            if self.confidence_recorder is not None and proposal["confidence_logits"] is not None:
                self.confidence_recorder.observe(
                    confidence_logits=proposal["confidence_logits"],
                    accept_prefix_mask=accept_prefix_mask,
                    effective_length=effective_proposal_length,
                )

            proposal_lengths.append(effective_proposal_length)
            accepted_draft_lengths.append(accepted)
            output_ids[:, start : start + accepted + 1] = proposal["verify_input_ids"][
                :, : accepted + 1
            ]

            if terminated_by_stop:
                acceptance_lengths.append(accepted)
                start += accepted
                _crop_cache(past_key_values_target, start)
                break

            output_ids[:, start + accepted + 1] = next_token
            new_token_ids = output_ids[:, start + 1 : start + accepted + 2]
            acceptance_lengths.append(accepted + 1)
            start += accepted + 1
            _crop_cache(past_key_values_target, start)

            target_hidden = extract_context_feature(
                target_output.hidden_states, draft.target_layer_ids
            )[:, : accepted + 1, :]
            if has_stop_token(new_token_ids, self.stop_token_ids):
                break

        output_ids = output_ids[:, : min(start + 1, max_length)]
        output_ids = trim_stop_tokens(output_ids, num_input_tokens, self.stop_token_ids)
        return self._build_metrics(
            prompt,
            output_ids,
            num_input_tokens,
            proposal_lengths,
            accepted_draft_lengths,
            acceptance_lengths,
            total_time=time.perf_counter() - total_start,
            target_time=target_time,
            draft_time=draft_time,
        )

    def _propose(
        self,
        *,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values_draft: DynamicCache,
        target_hidden: torch.Tensor,
        start: int,
        block_size: int,
        mask_token_id: int,
    ) -> dict[str, Any]:
        draft = self.draft_model
        draft_input_ids = torch.full(
            (output_ids.size(0), block_size),
            mask_token_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        noise_embedding = self.embed_tokens(draft_input_ids).to(dtype=target_hidden.dtype)

        t0 = time.perf_counter()
        block_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, :block_size, :]
        _crop_cache(past_key_values_draft, start)
        draft_time = time.perf_counter() - t0

        base_logits = self.lm_head(block_hidden)
        sampled_tokens, draft_logits = self._sample_dspark_block(
            base_logits=base_logits,
            first_prev_token_ids=draft_input_ids[:, 0],
            hidden_states=block_hidden,
        )
        proposal_len = block_size
        confidence_logits = self._predict_confidence(
            block_hidden=block_hidden,
            draft_input_ids=draft_input_ids,
            sampled_tokens=sampled_tokens,
        )
        if confidence_logits is not None and self.confidence_threshold > 0.0:
            below = confidence_logits.sigmoid() < self.confidence_threshold
            if bool(below[0].any().item()):
                proposal_len = int(torch.nonzero(below[0], as_tuple=False)[0].item())
        if proposal_len == 0:
            return {
                "draft_token_count": 0,
                "verify_input_ids": draft_input_ids[:, :1],
                "draft_probs": None,
                "confidence_logits": confidence_logits,
                "draft_time_sec": draft_time,
            }
        return {
            "draft_token_count": proposal_len,
            "verify_input_ids": torch.cat(
                [draft_input_ids[:, :1], sampled_tokens[:, :proposal_len]], dim=1
            ),
            "draft_probs": logits_to_probs(draft_logits[:, :proposal_len, :], self.temperature),
            "confidence_logits": confidence_logits,
            "draft_time_sec": draft_time,
        }

    def _sample_dspark_block(
        self,
        *,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draft = self.draft_model
        sampled = []
        logits = []
        prev = first_prev_token_ids
        for pos in range(base_logits.size(1)):
            step_logits = base_logits[:, pos, :]
            if draft.markov_head is not None:
                step_logits = step_logits + draft.markov_head.compute_step_bias(prev)
            token = sample_tokens(step_logits.unsqueeze(1), self.temperature).squeeze(1)
            sampled.append(token)
            logits.append(step_logits)
            prev = token
        return torch.stack(sampled, dim=1), torch.stack(logits, dim=1)

    def _predict_confidence(
        self,
        *,
        block_hidden: torch.Tensor,
        draft_input_ids: torch.Tensor,
        sampled_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        draft = self.draft_model
        if draft.confidence_head is None:
            return None
        if draft.confidence_head_with_markov:
            prev_token_ids = torch.cat([draft_input_ids[:, :1], sampled_tokens[:, :-1]], dim=1)
            prev_emb = draft.markov_head.get_prev_embeddings(prev_token_ids).to(block_hidden.dtype)
            features = torch.cat([block_hidden, prev_emb], dim=-1)
        else:
            features = block_hidden
        return draft.confidence_head(features).float()

    def _verify(
        self,
        *,
        verify_input_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values_target: Any,
        start: int,
        draft_token_count: int,
        block_size: int,
        confidence_logits: torch.Tensor | None,
    ) -> dict[str, Any]:
        verify_len = draft_token_count + 1
        target_output = self.target_model(
            verify_input_ids,
            position_ids=position_ids[:, start : start + verify_len],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        target_logits = target_output.logits
        target_probs = logits_to_probs(target_logits, self.temperature)

        accept_prefix_mask = None
        accepted = 0
        if draft_token_count > 0:
            proposed = verify_input_ids[:, 1:]
            if self.temperature < 1e-5:
                target_tokens = torch.argmax(target_logits[:, :-1, :], dim=-1)
                accept_mask = (proposed == target_tokens).to(torch.int64)
            else:
                assert draft_probs is not None
                selected_target_probs = gather_token_probs(target_probs[:, :-1, :], proposed)
                selected_draft_probs = gather_token_probs(draft_probs, proposed).clamp_min(1e-8)
                accept_prob = torch.clamp(selected_target_probs / selected_draft_probs, max=1.0)
                accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
            accept_prefix_mask = accept_mask.cumprod(dim=1)
            accepted = int(accept_prefix_mask.sum(dim=1)[0].item())

        effective_proposal_length = draft_token_count
        terminated_by_stop = False
        if self.stop_token_ids and accepted > 0:
            accepted_slice = verify_input_ids[0, 1 : accepted + 1]
            stop = torch.tensor(
                self.stop_token_ids,
                device=accepted_slice.device,
                dtype=accepted_slice.dtype,
            )
            eos_hits = torch.isin(accepted_slice, stop).nonzero(as_tuple=True)[0]
            if eos_hits.numel() > 0:
                accepted = int(eos_hits[0].item()) + 1
                effective_proposal_length = accepted
                terminated_by_stop = True
                if accept_prefix_mask is not None:
                    accept_prefix_mask = accept_prefix_mask[:, :accepted]

        if 0 < draft_token_count and accepted < draft_token_count:
            assert draft_probs is not None
            if self.temperature < 1e-5:
                next_token = torch.argmax(target_logits[:, accepted, :], dim=-1)
            else:
                next_token = sample_residual(
                    target_probs[:, accepted, :], draft_probs[:, accepted, :]
                )
        else:
            next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

        if accept_prefix_mask is None:
            accept_prefix_mask = torch.ones(
                (verify_input_ids.size(0), max(accepted, 1)),
                dtype=torch.int64,
                device=verify_input_ids.device,
            )

        return {
            "target_output": target_output,
            "past_key_values_target": target_output.past_key_values,
            "accept_prefix_mask": accept_prefix_mask,
            "accepted_draft_tokens": accepted,
            "next_token": next_token,
            "effective_proposal_length": effective_proposal_length,
            "terminated_by_stop": terminated_by_stop,
        }

    def _build_metrics(
        self,
        prompt: str,
        output_ids: torch.Tensor,
        num_input_tokens: int,
        proposal_lengths: list[int],
        accepted_draft_lengths: list[int],
        acceptance_lengths: list[int],
        total_time: float,
        target_time: float,
        draft_time: float,
    ) -> SampleMetrics:
        output_text = self.tokenizer.decode(
            output_ids[0, num_input_tokens:], skip_special_tokens=True
        )
        return SampleMetrics(
            prompt=prompt,
            output=output_text,
            num_input_tokens=num_input_tokens,
            num_output_tokens=max(int(output_ids.shape[1] - num_input_tokens), 0),
            verify_count=len(proposal_lengths),
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            acceptance_lengths=acceptance_lengths,
            total_time_sec=total_time,
            target_time_sec=target_time,
            draft_time_sec=draft_time,
        )


def parse_benchmark_item(item: str) -> tuple[str, int | None, list[str] | None]:
    parts = item.split(":")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], int(parts[1]), None
    if len(parts) == 3:
        return parts[0], int(parts[1]), parts[2].split(",")
    raise ValueError(f"Invalid benchmark item: {item}")


def question_to_prompt(question: dict[str, Any], benchmarker: Any) -> str:
    if "question" in question:
        prompt = str(question["question"])
        few_shot = getattr(benchmarker, "few_shot_examples", None)
        return f"{few_shot}{prompt}" if few_shot else prompt
    if "question_1" in question:
        return str(question["question_1"])
    return "\n".join(str(value) for value in question.values())


def normalize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role", message.get("from", ""))
    content = message.get("content") or message.get("value") or ""
    if role in ("human", "user"):
        role = "user"
    elif role in ("gpt", "assistant"):
        role = "assistant"
    return {"role": role, "content": content}


def render_jsonl_prompt(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    is_preformatted: bool,
) -> str:
    if is_preformatted:
        if "text" not in row:
            raise ValueError(
                f"Expected 'text' field for --is-preformatted, got keys: {list(row.keys())}"
            )
        text = str(row["text"])
        assistant_header = template.assistant_header or ""
        marker = text.rfind(assistant_header) if assistant_header else -1
        if marker < 0:
            end_marker = template.end_of_turn_token or ""
            marker = text.rfind(end_marker) if end_marker else -1
        return text[:marker] if marker >= 0 else text

    if "conversations" not in row:
        raise ValueError(f"Expected 'conversations' field, got keys: {list(row.keys())}")
    messages = [normalize_chat_message(message) for message in row["conversations"]]
    while messages and messages[-1].get("role") == "assistant":
        messages.pop()
    while messages and messages[0].get("role") not in ("system", "user"):
        messages.pop(0)
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("No user message found in conversations.")
    if template.system_prompt and messages and messages[0].get("role") != "system":
        messages = [{"role": "system", "content": template.system_prompt}] + messages
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=getattr(template, "enable_thinking", False),
    )


def load_jsonl_prompts(
    path: str,
    *,
    tokenizer: Any,
    chat_template: str,
    is_preformatted: bool,
    max_length: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    template = TEMPLATE_REGISTRY.get(chat_template)
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break

    prompts = []
    for index, row in enumerate(rows):
        prompt_text = render_jsonl_prompt(
            row,
            tokenizer=tokenizer,
            template=template,
            is_preformatted=is_preformatted,
        )
        input_ids = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        prompts.append(
            {
                "prompt": row.get("id", f"{os.path.basename(path)}:{index}"),
                "input_ids": input_ids,
            }
        )
    return prompts


def load_eval_sets(args: argparse.Namespace, tokenizer: Any) -> list[tuple[str, list[Any], Any, Any]]:
    eval_sets = []
    if args.prompt:
        eval_sets.append(("prompt", [args.prompt], None, None))
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        eval_sets.append((os.path.basename(args.prompts_file), prompts, None, None))
    if args.eval_data_path:
        prompts = load_jsonl_prompts(
            args.eval_data_path,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            is_preformatted=args.is_preformatted,
            max_length=args.max_input_length,
            limit=args.eval_data_limit,
        )
        eval_sets.append((os.path.basename(args.eval_data_path), prompts, None, None))
    for item in args.benchmark_list:
        from benchmarker import BENCHMARKS

        name, num_samples, subset = parse_benchmark_item(item)
        cls = BENCHMARKS.get(name)
        benchmarker = cls(num_samples=num_samples, subset=subset) if subset else cls(num_samples=num_samples)
        questions, labels = benchmarker.load_data()
        prompts = [question_to_prompt(question, benchmarker) for question in questions]
        eval_sets.append((name, prompts, labels, benchmarker))

    if args.num_samples is not None:
        eval_sets = [
            (name, prompts[: args.num_samples], labels, benchmarker)
            for name, prompts, labels, benchmarker in eval_sets
        ]

    if not eval_sets:
        raise ValueError("Provide --prompt, --prompts-file, --eval-data-path, or --benchmark-list.")
    return eval_sets


def encode_prompt(tokenizer: Any, prompt: str, apply_chat_template: bool) -> torch.Tensor:
    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def _aggregate_samples(
    samples: list[SampleMetrics],
    block_size: int,
) -> _DatasetAggregate:
    proposal_lengths = [value for sample in samples for value in sample.proposal_lengths]
    accepted_lengths = [value for sample in samples for value in sample.accepted_draft_lengths]
    acceptance_lengths = [value for sample in samples for value in sample.acceptance_lengths]

    proposals_at_pos = [0] * block_size
    accepted_at_pos = [0] * block_size
    for proposal_length, accepted_length in zip(proposal_lengths, accepted_lengths):
        for pos in range(block_size):
            if proposal_length > pos:
                proposals_at_pos[pos] += 1
            if accepted_length > pos:
                accepted_at_pos[pos] += 1

    return _DatasetAggregate(
        sample_count=len(samples),
        proposal_count=len(proposal_lengths),
        acceptance_length_sum=sum(acceptance_lengths),
        proposal_length_sum=sum(proposal_lengths),
        proposals_at_pos=proposals_at_pos,
        accepted_at_pos=accepted_at_pos,
        output_token_sum=sum(sample.num_output_tokens for sample in samples),
        verify_count=sum(sample.verify_count for sample in samples),
        total_time_sec=sum(sample.total_time_sec for sample in samples),
        target_time_sec=sum(sample.target_time_sec for sample in samples),
        draft_time_sec=sum(sample.draft_time_sec for sample in samples),
    )


def _all_reduce_aggregate(aggregate: _DatasetAggregate, device: torch.device) -> _DatasetAggregate:
    """All-reduce dataset counters across ranks in place."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return aggregate

    scalar = torch.tensor(
        [
            aggregate.sample_count,
            aggregate.proposal_count,
            aggregate.acceptance_length_sum,
            aggregate.proposal_length_sum,
            aggregate.output_token_sum,
            aggregate.verify_count,
        ],
        dtype=torch.int64,
        device=device,
    )
    # NPU collectives do not support float64; mirror ConfidenceRecorder logic.
    time_dtype = torch.float32 if device.type == "npu" else torch.float64
    float_scalar = torch.tensor(
        [aggregate.total_time_sec, aggregate.target_time_sec, aggregate.draft_time_sec],
        dtype=time_dtype,
        device=device,
    )
    pos_counts = torch.tensor(
        aggregate.proposals_at_pos + aggregate.accepted_at_pos,
        dtype=torch.int64,
        device=device,
    )

    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    dist.all_reduce(float_scalar, op=dist.ReduceOp.SUM)
    dist.all_reduce(pos_counts, op=dist.ReduceOp.SUM)

    block_size = len(aggregate.proposals_at_pos)
    return _DatasetAggregate(
        sample_count=int(scalar[0].item()),
        proposal_count=int(scalar[1].item()),
        acceptance_length_sum=int(scalar[2].item()),
        proposal_length_sum=int(scalar[3].item()),
        proposals_at_pos=[int(v) for v in pos_counts[:block_size].tolist()],
        accepted_at_pos=[int(v) for v in pos_counts[block_size:].tolist()],
        output_token_sum=int(scalar[4].item()),
        verify_count=int(scalar[5].item()),
        total_time_sec=float(float_scalar[0].item()),
        target_time_sec=float(float_scalar[1].item()),
        draft_time_sec=float(float_scalar[2].item()),
    )


def _build_dataset_metrics(
    dataset: str,
    aggregate: _DatasetAggregate,
    block_size: int,
    accuracy: float | None,
) -> DatasetMetrics:
    proposal_count = aggregate.proposal_count
    if proposal_count == 0:
        draft_tokens_per_proposal = 0.0
        acceptance_length = 0.0
        verify_rate = 0.0
        accept_rates_by_position = [None] * block_size
    else:
        acceptance_length = aggregate.acceptance_length_sum / proposal_count
        draft_tokens_per_proposal = aggregate.proposal_length_sum / proposal_count
        verify_rate = aggregate.acceptance_length_sum / (
            aggregate.proposal_length_sum + proposal_count
        )
        accept_rates_by_position = []
        for pos in range(block_size):
            denom = aggregate.proposals_at_pos[pos]
            if denom == 0:
                accept_rates_by_position.append(None)
            else:
                accept_rates_by_position.append(
                    aggregate.accepted_at_pos[pos] / denom
                )

    total_time = aggregate.total_time_sec
    output_tokens = aggregate.output_token_sum
    tokens_per_sec = output_tokens / total_time if total_time > 1e-6 else 0.0

    return DatasetMetrics(
        dataset=dataset,
        num_samples=aggregate.sample_count,
        draft_tokens_per_proposal=draft_tokens_per_proposal,
        acceptance_length=acceptance_length,
        verify_rate=verify_rate,
        accept_rates_by_position=accept_rates_by_position,
        accuracy=accuracy,
        total_time_sec=total_time,
        tokens_per_sec=tokens_per_sec,
        target_time_sec=aggregate.target_time_sec,
        draft_time_sec=aggregate.draft_time_sec,
    )


def _gather_sample_rows(
    samples: list[SampleMetrics],
    world_size: int,
    total_prompts: int,
) -> list[SampleMetrics]:
    """Gather sample metrics on rank 0 and restore original global order."""
    if world_size <= 1:
        return samples

    gathered: list[list[SampleMetrics] | None] = [None] * world_size
    dist.all_gather_object(gathered, samples)

    if dist.get_rank() != 0:
        return []

    ordered: list[SampleMetrics] = []
    for global_idx in range(total_prompts):
        rank = global_idx % world_size
        local_idx = global_idx // world_size
        ordered.append(gathered[rank][local_idx])
    return ordered


def summarize_dataset(
    dataset: str,
    samples: list[SampleMetrics],
    labels: Any,
    benchmarker: Any,
    block_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    total_prompts: int,
) -> DatasetMetrics:
    aggregate = _aggregate_samples(samples, block_size)
    aggregate = _all_reduce_aggregate(aggregate, device)

    all_samples = samples
    if labels is not None and benchmarker is not None and world_size > 1:
        all_samples = _gather_sample_rows(samples, world_size, total_prompts)

    accuracy = None
    if labels is not None and benchmarker is not None and rank == 0:
        predictions = [
            benchmarker.extract_answer(sample.output, labels[idx])
            for idx, sample in enumerate(all_samples)
        ]
        accuracy = benchmarker.compute_accuracy(predictions, labels)

    return _build_dataset_metrics(dataset, aggregate, block_size, accuracy)


def _shard_prompts(prompts: list[Any], rank: int, world_size: int) -> list[Any]:
    if world_size <= 1:
        return prompts
    return [prompt for idx, prompt in enumerate(prompts) if idx % world_size == rank]


def main() -> None:
    args = parse_args()
    device, rank, world_size = init_distributed(args.device, args.distributed_backend)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attention_backend,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    draft_model = DSparkDraftModel.from_pretrained(
        args.draft_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attention_backend,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    draft_model.config._attn_implementation = args.attention_backend

    assert_no_final_target_layer(target_model, draft_model.target_layer_ids)
    assert 0.0 <= float(args.confidence_threshold) <= 1.0

    confidence_recorder: ConfidenceRecorder | None = None
    if (
        draft_model.confidence_head is not None
        and float(args.confidence_threshold) == 0.0
    ):
        confidence_recorder = ConfidenceRecorder(
            block_size=int(draft_model.block_size),
            device=device,
        )

    generator = DSparkHFGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        max_new_tokens=args.max_new_tokens,
        confidence_recorder=confidence_recorder,
    )

    results = {
        "target_model": args.target_model_path,
        "draft_model": args.draft_model_path,
        "world_size": world_size,
        "datasets": [],
    }

    for dataset_name, prompts, labels, benchmarker in load_eval_sets(args, tokenizer):
        local_prompts = _shard_prompts(prompts, rank, world_size)
        if rank == 0:
            print(
                f"Running {dataset_name} with {len(prompts)} prompts "
                f"(local shard {len(local_prompts)} on rank {rank})"
            )

        sample_rows = []
        for index, prompt_item in enumerate(local_prompts, start=1):
            if isinstance(prompt_item, dict):
                prompt = str(prompt_item["prompt"])
                input_ids = prompt_item["input_ids"]
            else:
                prompt = str(prompt_item)
                input_ids = encode_prompt(tokenizer, prompt, args.apply_chat_template)
            metrics = generator.generate(input_ids, prompt)
            sample_rows.append(metrics)
            print(
                f"[{dataset_name} r{rank} {index}/{len(local_prompts)}] "
                f"out={metrics.num_output_tokens} verify={metrics.verify_count} "
                f"accept_len={(statistics.mean(metrics.acceptance_lengths) if metrics.acceptance_lengths else 0.0):.2f} "
                f"time={metrics.total_time_sec:.3f}s"
            )

        if confidence_recorder is not None:
            confidence_recorder.all_reduce()

        summary = summarize_dataset(
            dataset_name,
            sample_rows,
            labels,
            benchmarker,
            int(draft_model.block_size),
            device,
            rank,
            world_size,
            len(prompts),
        )

        confidence_summary = None
        if confidence_recorder is not None and rank == 0:
            confidence_summary = confidence_recorder.compute()
            confidence_recorder.reset()

        if rank == 0:
            summary_dict = asdict(summary)
            if confidence_summary is not None:
                summary_dict["confidence_calibration"] = confidence_summary
            print(json.dumps(summary_dict, indent=2, ensure_ascii=False))
            results["datasets"].append(
                {"summary": summary_dict, "samples": [asdict(row) for row in sample_rows]}
            )

    if args.output_file and rank == 0:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
