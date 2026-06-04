"""paper_explore SYS22 — per-request aggregate logprob accumulator.

Used by the inline cascade-routing decider. When a request opts in via
`SamplingParams.emit_aggregate_logprob_stats=True`, gpu_model_runner
updates this accumulator each decode step from the sampler's per-step
top-K logprob distribution. At request finalization, the 4 final
scalars land in `CompletionOutput.aggregate_logprob_stats`.

The 4 scalars:
  mean_logprob:     mean chosen-token logprob across the generation
  min_logprob:      minimum chosen-token logprob (worst-confidence
                    single token)
  mean_max_prob:    mean of max-softmax across decoded steps (over
                    top-K)
  neg_mean_entropy: -mean of top-K predictive entropy across steps
                    (higher = more confident, less entropy)

Same semantics as the existing driver-side aggregation in
paper_explore/glue/engine.py (DraftEngineAsync._drive's logprob stats
computation), but accumulated inline on the GPU so per-token logprob
dicts don't need to be serialized to CPU.

Driver-side equivalence test (in paper_explore CI):
  same(stats_from_inline_accum,
       paper_explore.glue.engine.aggregate_4_from_per_token_logprobs)
must hold to within fp32 precision over a corpus of test sequences.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PerRequestLogprobStats:
    """One per active request. Trivial memory (5 floats / 1 int)."""

    sum_chosen_logprob: float = 0.0
    min_chosen_logprob: float = float("inf")
    sum_max_prob: float = 0.0
    sum_neg_entropy: float = 0.0   # accumulates -entropy (so larger
                                    # is more confident)
    n_steps: int = 0

    def update_from_step(
        self,
        chosen_logprob: float,
        topk_logprobs: torch.Tensor,  # [K] sorted descending
    ) -> None:
        """Update from one decode step's outputs."""
        self.sum_chosen_logprob += chosen_logprob
        if chosen_logprob < self.min_chosen_logprob:
            self.min_chosen_logprob = chosen_logprob
        # max_prob: top-K is sorted desc, so first entry is max
        max_prob = float(torch.exp(topk_logprobs[0]))
        self.sum_max_prob += max_prob
        # top-K entropy: -sum (p/z) log (p/z) where p = exp(logprob)
        # Renormalize over the top-K to get a proper distribution
        # (since we truncated the tail). Standard approx for full
        # predictive entropy.
        probs = torch.exp(topk_logprobs)
        z = probs.sum()
        if float(z) > 0 and topk_logprobs.shape[0] > 1:
            normalized = probs / z
            # Skip zeros (shouldn't happen with exp, but be safe).
            log_p = torch.log(normalized.clamp(min=1e-30))
            entropy = -float((normalized * log_p).sum())
            self.sum_neg_entropy += -entropy  # negate so larger = more confident
        self.n_steps += 1

    def finalize(self) -> dict[str, float] | None:
        """Read out the 4 aggregate stats. None if no steps."""
        n = self.n_steps
        if n == 0:
            return None
        return {
            "mean_logprob": self.sum_chosen_logprob / n,
            "min_logprob": self.min_chosen_logprob,
            "mean_max_prob": self.sum_max_prob / n,
            "neg_mean_entropy": self.sum_neg_entropy / n,
        }


class LogprobStatsAccumulator:
    """Per-engine registry of per-request accumulators.

    Lifecycle:
      - register_request(req_id): called when a request enters the
        running set with emit_aggregate_logprob_stats=True.
      - update(req_id, chosen_lp, topk_lp): called each decode step
        from gpu_model_runner._bookkeeping_sync.
      - finalize(req_id): called when request finishes; returns the
        4-stat dict to attach to CompletionOutput.

    Memory: 5 floats + 1 int per active request. Negligible.
    """

    def __init__(self) -> None:
        self._state: dict[str, PerRequestLogprobStats] = {}

    def register_request(self, req_id: str) -> None:
        self._state.setdefault(req_id, PerRequestLogprobStats())

    def has_request(self, req_id: str) -> bool:
        return req_id in self._state

    def update(
        self,
        req_id: str,
        chosen_logprob: float,
        topk_logprobs: torch.Tensor,
    ) -> None:
        st = self._state.get(req_id)
        if st is None:
            # Request didn't opt in (or already finalized); skip.
            return
        st.update_from_step(chosen_logprob, topk_logprobs)

    def finalize(self, req_id: str) -> dict[str, float] | None:
        st = self._state.pop(req_id, None)
        if st is None:
            return None
        return st.finalize()


# ---------------------------------------------------------------------
# Hook called from gpu_model_runner._bookkeeping_sync after the sampler
# has produced logprobs_tensors. NOT YET INTEGRATED into the model
# runner — see PAPER_EXPLORE_LP_CLASSIFIER_INLINE.md for the rest.
# ---------------------------------------------------------------------

def update_accumulator_for_step(
    accumulator: LogprobStatsAccumulator,
    req_ids: list[str],
    opted_in_mask: list[bool],
    sampler_output_logprobs: Any,  # vLLM SamplerOutput.logprobs_tensors
) -> None:
    """Pseudocode for the per-step hook. Called in
    gpu_model_runner._bookkeeping_sync immediately after the sampler.

    Real integration needs to:
      1. Determine for each request whether it opted in (from input_batch
         flag tracked since request submission).
      2. For each opted-in request, extract chosen-token logprob + top-K
         logprobs at the step's position in the LogprobsTensors flat
         layout (indexing by req_idx * max_gen_len + step_idx).
      3. Call accumulator.update(req_id, chosen_lp, topk_lp).
    """
    # See PAPER_EXPLORE_LP_CLASSIFIER_INLINE.md for the integration
    # details and the validation test plan. This stub exists to mark
    # the integration point and document the expected interface.
    raise NotImplementedError(
        "update_accumulator_for_step: integration into gpu_model_runner."
        "_bookkeeping_sync is still TODO. See "
        "PAPER_EXPLORE_LP_CLASSIFIER_INLINE.md for the design."
    )
