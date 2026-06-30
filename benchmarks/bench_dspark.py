#!/usr/bin/env python3
"""Run HF-only DSpark speculative-decoding evaluation.

This script evaluates a SpecForge-trained DSpark draft checkpoint directly, without
converting it to DeepSpec format or launching an SGLang server.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from specforge.modeling.draft.dflash import extract_context_feature
from specforge.modeling.draft.dspark import DSparkDraftModel


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


@dataclass
class DatasetMetrics:
    dataset: str
    num_samples: int
    draft_tokens_per_proposal: float
    acceptance_length: float
    verify_rate: float
    accept_rates_by_position: list[float | None]
    accuracy: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DSpark draft checkpoint")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--benchmark-list", nargs="*", default=[])
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
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax((logits / temperature).float(), dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(logits.shape[:-1])


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scale = max(float(temperature), 1e-5)
    return torch.softmax((logits / scale).float(), dim=-1)


def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    residual = (target_probs - draft_probs).clamp_min(0.0)
    denom = residual.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 1e-8, residual / denom.clamp_min(1e-8), target_probs)
    return torch.multinomial(probs, 1).squeeze(-1)


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

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, prompt: str) -> SampleMetrics:
        target = self.target_model
        draft = self.draft_model
        input_ids = input_ids.to(target.device)
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + self.max_new_tokens
        block_size = int(draft.block_size)
        mask_token_id = int(draft.mask_token_id)

        output_ids = torch.full(
            (1, max_length + block_size + 1),
            mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        initial = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
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
            return self._build_metrics(prompt, output_ids, num_input_tokens, [], [], [])

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
            draft_token_count = proposal["draft_token_count"]
            verify_input_ids = proposal["verify_input_ids"]
            draft_probs = proposal["draft_probs"]

            verification = self._verify(
                verify_input_ids=verify_input_ids,
                draft_probs=draft_probs,
                position_ids=position_ids,
                past_key_values_target=past_key_values_target,
                start=start,
                draft_token_count=draft_token_count,
                block_size=block_size,
            )

            accepted = verification["accepted_draft_tokens"]
            next_token = verification["next_token"]
            terminated_by_stop = verification["terminated_by_stop"]
            target_output = verification["target_output"]
            effective_proposal_length = verification["effective_proposal_length"]

            proposal_lengths.append(effective_proposal_length)
            accepted_draft_lengths.append(accepted)
            output_ids[:, start : start + accepted + 1] = verify_input_ids[:, : accepted + 1]

            if terminated_by_stop:
                acceptance_lengths.append(accepted)
                start += accepted
                past_key_values_target.crop(start)
                break

            output_ids[:, start + accepted + 1] = next_token
            new_token_ids = output_ids[:, start + 1 : start + accepted + 2]
            acceptance_lengths.append(accepted + 1)
            start += accepted + 1
            past_key_values_target.crop(start)

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
        past_key_values_draft.crop(start)

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
            }
        return {
            "draft_token_count": proposal_len,
            "verify_input_ids": torch.cat(
                [draft_input_ids[:, :1], sampled_tokens[:, :proposal_len]], dim=1
            ),
            "draft_probs": logits_to_probs(draft_logits[:, :proposal_len, :], self.temperature),
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
        past_key_values_target: DynamicCache,
        start: int,
        draft_token_count: int,
        block_size: int,
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

        accepted = 0
        if draft_token_count > 0:
            proposed = verify_input_ids[:, 1:]
            if self.temperature <= 1e-5:
                target_tokens = torch.argmax(target_logits[:, :-1, :], dim=-1)
                accepted = int(((proposed == target_tokens).cumprod(dim=1)).sum(dim=1)[0].item())
            else:
                assert draft_probs is not None
                selected_target_probs = gather_token_probs(target_probs[:, :-1, :], proposed)
                selected_draft_probs = gather_token_probs(draft_probs, proposed).clamp_min(1e-8)
                accept_prob = torch.clamp(selected_target_probs / selected_draft_probs, max=1.0)
                accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
                accepted = int(accept_mask.cumprod(dim=1).sum(dim=1)[0].item())

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

        if 0 < draft_token_count and accepted < draft_token_count:
            assert draft_probs is not None
            if self.temperature <= 1e-5:
                next_token = torch.argmax(target_logits[:, accepted, :], dim=-1)
            else:
                next_token = sample_residual(
                    target_probs[:, accepted, :], draft_probs[:, accepted, :]
                )
        else:
            next_token = sample_tokens(target_logits[:, -1:, :], self.temperature).squeeze(1)

        return {
            "target_output": target_output,
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


def load_eval_sets(args: argparse.Namespace) -> list[tuple[str, list[str], Any, Any]]:
    eval_sets = []
    if args.prompt:
        eval_sets.append(("prompt", [args.prompt], None, None))
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        eval_sets.append((os.path.basename(args.prompts_file), prompts, None, None))
    for item in args.benchmark_list:
        from benchmarker import BENCHMARKS

        name, num_samples, subset = parse_benchmark_item(item)
        cls = BENCHMARKS.get(name)
        benchmarker = cls(num_samples=num_samples, subset=subset) if subset else cls(num_samples=num_samples)
        questions, labels = benchmarker.load_data()
        prompts = [question_to_prompt(question, benchmarker) for question in questions]
        eval_sets.append((name, prompts, labels, benchmarker))
    if not eval_sets:
        raise ValueError("Provide --prompt, --prompts-file, or --benchmark-list.")
    return eval_sets


def encode_prompt(tokenizer: Any, prompt: str, apply_chat_template: bool) -> torch.Tensor:
    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def summarize_dataset(
    dataset: str,
    samples: list[SampleMetrics],
    labels: Any,
    benchmarker: Any,
    block_size: int,
) -> DatasetMetrics:
    proposal_lengths = [value for sample in samples for value in sample.proposal_lengths]
    accepted_lengths = [value for sample in samples for value in sample.accepted_draft_lengths]
    acceptance_lengths = [value for sample in samples for value in sample.acceptance_lengths]
    num_output_tokens = sum(sample.num_output_tokens for sample in samples)
    verify_count = sum(sample.verify_count for sample in samples)

    accept_rates = []
    for pos in range(block_size):
        denom = sum(1 for length in proposal_lengths if length > pos)
        if denom == 0:
            accept_rates.append(None)
        else:
            numer = sum(1 for length in accepted_lengths if length > pos)
            accept_rates.append(numer / denom)

    accuracy = None
    if labels is not None and benchmarker is not None:
        predictions = [benchmarker.extract_answer(sample.output, labels[idx]) for idx, sample in enumerate(samples)]
        accuracy = benchmarker.compute_accuracy(predictions, labels)

    return DatasetMetrics(
        dataset=dataset,
        num_samples=len(samples),
        draft_tokens_per_proposal=(statistics.mean(proposal_lengths) if proposal_lengths else 0.0),
        acceptance_length=(statistics.mean(acceptance_lengths) if acceptance_lengths else 0.0),
        verify_rate=(verify_count / max(num_output_tokens, 1)),
        accept_rates_by_position=accept_rates,
        accuracy=accuracy,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype=dtype,
        attn_implementation=args.attention_backend,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    draft_model = DSparkDraftModel.from_pretrained(
        args.draft_model_path,
        dtype=dtype,
        attn_implementation=args.attention_backend,
        trust_remote_code=True,
    ).to(device).eval()
    draft_model.config._attn_implementation = args.attention_backend

    generator = DSparkHFGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        max_new_tokens=args.max_new_tokens,
    )

    results = {"target_model": args.target_model_path, "draft_model": args.draft_model_path, "datasets": []}
    for dataset_name, prompts, labels, benchmarker in load_eval_sets(args):
        print(f"Running {dataset_name} with {len(prompts)} prompts")
        sample_rows = []
        for index, prompt in enumerate(prompts, start=1):
            input_ids = encode_prompt(tokenizer, prompt, args.apply_chat_template)
            metrics = generator.generate(input_ids, prompt)
            sample_rows.append(metrics)
            print(
                f"[{dataset_name} {index}/{len(prompts)}] "
                f"out={metrics.num_output_tokens} verify={metrics.verify_count} "
                f"accept_len={(statistics.mean(metrics.acceptance_lengths) if metrics.acceptance_lengths else 0.0):.2f}"
            )
        summary = summarize_dataset(
            dataset_name, sample_rows, labels, benchmarker, int(draft_model.block_size)
        )
        print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
        results["datasets"].append(
            {"summary": asdict(summary), "samples": [asdict(row) for row in sample_rows]}
        )

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
