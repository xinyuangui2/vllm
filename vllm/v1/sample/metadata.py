# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None

    # paper_explore SYS25 Phase 3 — True if any request in the batch
    # opted in via SamplingParams.emit_per_token_feature_seq. When set,
    # the sampler computes [chosen_lp, max_p, neg_entropy] inline from
    # log_softmax(logits) and emits SamplerOutput.feature_seq_tensor.
    # Independent of max_num_logprobs so feature_seq can run without
    # paying for gather_logprobs (topk + rank).
    has_emit_per_token_feature_seq: bool = False
