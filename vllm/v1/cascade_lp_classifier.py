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
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.logprobs import SampleLogprobs


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
# SYS25 — per-token feature seq accumulator.
#
# Extension of the aggregate path: instead of summing per-step features
# into 5 scalars, store one row of [chosen_lp, max_p, neg_entropy] per
# decode step. At finalization, the driver tacks on pos_frac (= (i+1)/T)
# and returns the [T, 4] feature array to be consumed directly by the
# scheduler-side per-token gate (attn_pool / transformer_L2 in SYS22-T
# P18). Saves the per-step Logprob dict construction + driver-side
# detokenization that the standard `logprobs=K` path pays.
# ---------------------------------------------------------------------

@dataclass
class PerRequestFeatureSeq:
    """One per active request. Per-step rows of [chosen_lp, max_p,
    neg_entropy] (3 floats). pos_frac is added at finalize time."""

    chosen_lp: list[float] | None = None
    max_p: list[float] | None = None
    neg_entropy: list[float] | None = None

    def __post_init__(self) -> None:
        if self.chosen_lp is None:
            self.chosen_lp = []
        if self.max_p is None:
            self.max_p = []
        if self.neg_entropy is None:
            self.neg_entropy = []

    def update_from_step(
        self,
        chosen_logprob: float,
        topk_logprobs: torch.Tensor,  # [K] sorted descending
    ) -> None:
        """Append one row's (chosen_lp, max_p, neg_entropy). pos_frac
        is computed in finalize() since it depends on the final T."""
        self.chosen_lp.append(float(chosen_logprob))
        # max_prob: top-K is sorted desc; first entry is max
        self.max_p.append(float(torch.exp(topk_logprobs[0])))
        # top-K entropy: same math as PerRequestLogprobStats above
        probs = torch.exp(topk_logprobs)
        z = probs.sum()
        if float(z) > 0 and topk_logprobs.shape[0] > 1:
            normalized = probs / z
            log_p = torch.log(normalized.clamp(min=1e-30))
            entropy = -float((normalized * log_p).sum())
            self.neg_entropy.append(-entropy)
        else:
            self.neg_entropy.append(0.0)

    def finalize(self) -> list[list[float]] | None:
        """Return [T, 4] = [[chosen_lp, max_p, neg_entropy, pos_frac]]
        per row. None if no steps recorded."""
        T = len(self.chosen_lp)
        if T == 0:
            return None
        out = []
        for i in range(T):
            pos_frac = (i + 1) / T
            out.append([
                self.chosen_lp[i],
                self.max_p[i],
                self.neg_entropy[i],
                pos_frac,
            ])
        return out


class FeatureSeqAccumulator:
    """Per-engine registry of per-request feature-seq accumulators.

    Same lifecycle as LogprobStatsAccumulator (register/update/finalize)
    but stores per-step rows instead of running sums. Memory: ~3 floats
    per decoded token per active request — for 30 concurrent requests
    × 200 tokens = ~3 KB total, negligible.
    """

    def __init__(self) -> None:
        self._state: dict[str, PerRequestFeatureSeq] = {}

    def register_request(self, req_id: str) -> None:
        self._state.setdefault(req_id, PerRequestFeatureSeq())

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
            return
        st.update_from_step(chosen_logprob, topk_logprobs)

    def finalize(self, req_id: str) -> list[list[float]] | None:
        st = self._state.pop(req_id, None)
        if st is None:
            return None
        return st.finalize()


def compute_feature_seq_from_sample_logprobs(
    logprobs: "SampleLogprobs",
    token_ids: list[int],
) -> list[list[float]] | None:
    """Driver-side reference implementation of the feature seq.

    Same math as PerRequestFeatureSeq + pos_frac. Used as a numerical
    reference for validating the inline accumulator and as a fallback
    when the GPU side didn't produce features (e.g. partial generation).
    """
    if not logprobs or not token_ids:
        return None
    rows: list[list[float]] = []
    for pos, lp_dict in enumerate(logprobs):
        if lp_dict is None or pos >= len(token_ids):
            continue
        chosen_tid = token_ids[pos]
        chosen = lp_dict.get(chosen_tid)
        if chosen is None:
            continue
        chosen_lp = float(chosen.logprob)
        topk_lps = sorted(
            (float(v.logprob) for v in lp_dict.values()), reverse=True,
        )
        if not topk_lps:
            continue
        max_p = math.exp(topk_lps[0])
        neg_ent = 0.0
        if len(topk_lps) > 1:
            probs = [math.exp(lp) for lp in topk_lps]
            z = sum(probs)
            if z > 0:
                normalized = [p / z for p in probs]
                entropy = -sum(
                    p * math.log(max(p, 1e-30)) for p in normalized
                )
                neg_ent = -entropy
        rows.append([chosen_lp, max_p, neg_ent, 0.0])  # pos_frac filled below
    T = len(rows)
    if T == 0:
        return None
    for i in range(T):
        rows[i][3] = (i + 1) / T
    return rows


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


# ---------------------------------------------------------------------
# Driver-side reference implementation.
#
# Computes the same 4 stats as the inline accumulator, but from the
# already-serialized SampleLogprobs (per-position dict[token_id ->
# Logprob]). Same math, same outputs — just slower because it runs
# after CPU serialization rather than inline.
#
# Two purposes:
#   1. Numerical reference for validating the inline accumulator once
#      gpu_model_runner integration lands.
#   2. Lets paper_explore SYS22 cascade-routing exercise the new
#      SamplingParams + CompletionOutput field today, before the full
#      inline patch is wired up.
# ---------------------------------------------------------------------

def compute_aggregate_from_sample_logprobs(
    logprobs: "SampleLogprobs",
    token_ids: list[int],
) -> dict[str, float] | None:
    """Reference aggregator. Same math as PerRequestLogprobStats.

    logprobs[i] is dict[token_id -> Logprob] for the i-th generated
    position. token_ids[i] is the chosen token at that position.
    """
    if not logprobs or not token_ids:
        return None

    sum_chosen_lp = 0.0
    min_chosen_lp = float("inf")
    sum_max_prob = 0.0
    sum_neg_entropy = 0.0
    n_steps = 0

    for pos, lp_dict in enumerate(logprobs):
        if lp_dict is None or pos >= len(token_ids):
            continue
        chosen_tid = token_ids[pos]
        chosen = lp_dict.get(chosen_tid)
        if chosen is None:
            continue
        chosen_lp = float(chosen.logprob)
        sum_chosen_lp += chosen_lp
        if chosen_lp < min_chosen_lp:
            min_chosen_lp = chosen_lp

        # top-K logprobs at this position, sorted desc
        topk_lps = sorted(
            (float(v.logprob) for v in lp_dict.values()), reverse=True,
        )
        if not topk_lps:
            continue
        # mean_max_prob: top-1 softmax prob
        sum_max_prob += math.exp(topk_lps[0])

        # top-K entropy: renormalize over top-K
        if len(topk_lps) > 1:
            probs = [math.exp(lp) for lp in topk_lps]
            z = sum(probs)
            if z > 0:
                normalized = [p / z for p in probs]
                entropy = -sum(
                    p * math.log(max(p, 1e-30)) for p in normalized
                )
                sum_neg_entropy += -entropy

        n_steps += 1

    if n_steps == 0:
        return None
    return {
        "mean_logprob": sum_chosen_lp / n_steps,
        "min_logprob": min_chosen_lp,
        "mean_max_prob": sum_max_prob / n_steps,
        "neg_mean_entropy": sum_neg_entropy / n_steps,
    }


# ============================================================
# SYS25 Phase 4 / SYS25c — in-engine cascade head forward.
# Loaded once per OutputProcessor lifetime from env vars; fired
# at end-of-sequence from RequestState._new_completion_output.
# Brings the gate inside the engine so a single engine.generate
# call returns both the text and the SHIP/REGEN decision —
# eliminates the "engine emits logprobs → driver-side gate"
# split that the paper draft would otherwise have to explain.
#
# Arch dispatch (SYS25c): the same env-var path loads any of the
# trained SeqDataset architectures from
# scripts/sys22t_p16b_train_seq_models.py. Arch is read from
# ckpt["hparams"]["arch"]:
#   AttnPool       → _InEngineAttnPool      (Phase 4, ~449 params)
#   TransformerSeq → _InEngineTransformerL2 (SYS25c, ~67k params, L=2)
# Both mirror the driver-side module bit-exactly so loaded weights
# produce identical scores. Same `decide(feature_seq, source)` →
# CompletionOutput.head_decision interface either way.
# ============================================================


class _InEngineAttnPool(torch.nn.Module):
    """Mirror of paper_explore/scripts/sys22t_p16b_train_seq_models.py
    AttnPool. Replicated here so vLLM doesn't need to import the
    paper_explore module at engine init."""

    def __init__(self, n_features: int = 4, d_model: int = 64):
        super().__init__()
        self.proj = torch.nn.Linear(n_features, d_model)
        self.query = torch.nn.Parameter(torch.randn(d_model))
        self.classifier = torch.nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        scores = (h * self.query).sum(dim=-1)
        T = scores.shape[1]
        mask = (torch.arange(T, device=x.device).unsqueeze(0)
                < lens.to(x.device).unsqueeze(1))
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.nn.functional.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * attn).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


class _InEnginePositionalEncoding(torch.nn.Module):
    """Mirror of TransformerSeq's PositionalEncoding (sys22t_p16b).
    Registers the same `pe` buffer so a TransformerSeq state_dict
    loads with strict=True."""

    def __init__(self, d_model: int, max_len: int = 600):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class _InEngineTransformerL2(torch.nn.Module):
    """Mirror of paper_explore/scripts/sys22t_p16b_train_seq_models.py
    TransformerSeq. Same module names (`proj`, `pe`, `encoder`,
    `classifier`) and same torch.nn building blocks so loaded weights
    produce bit-identical scores to the driver-side gate.

    Default hparams = SYS22-T P18 transformer_L2:
      n_features=4, d_model=64, n_heads=4, n_layers=2, d_ff=128.
    """

    def __init__(self, n_features: int = 4, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, max_len: int = 600):
        super().__init__()
        self.proj = torch.nn.Linear(n_features, d_model)
        self.pe = _InEnginePositionalEncoding(d_model, max_len=max_len)
        enc = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier = torch.nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
               lens.to(x.device).unsqueeze(1)
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        not_mask = (~mask).float().unsqueeze(-1)
        pooled = (h * not_mask).sum(dim=1) / not_mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


def _build_head_module(hp: dict[str, Any]) -> torch.nn.Module:
    """Arch dispatch from ckpt["hparams"]. Keep this in sync with
    sys22t_p16b_train_seq_models.py's model_specs."""
    arch = hp.get("arch")
    if arch == "AttnPool":
        return _InEngineAttnPool(
            n_features=int(hp.get("n_features", 4)),
            d_model=int(hp.get("d_model", 64)),
        )
    if arch == "TransformerSeq":
        return _InEngineTransformerL2(
            n_features=int(hp.get("n_features", 4)),
            d_model=int(hp.get("d_model", 64)),
            n_heads=int(hp.get("n_heads", 4)),
            n_layers=int(hp.get("n_layers", 2)),
            d_ff=int(hp.get("d_ff", 128)),
            max_len=int(hp.get("max_len", 600)),
        )
    raise ValueError(
        f"InEngineCascadeHead: unsupported arch={arch!r}; expected "
        f"'AttnPool' or 'TransformerSeq' (see "
        f"sys22t_p16b_train_seq_models.py)."
    )


class InEngineCascadeHead:
    """In-engine cascade gate. Loaded once per OutputProcessor; decides
    SHIP/REGEN from the per-token feature seq at end of sequence.

    Decision: sigmoid(model(feature_seq)) >= τ[source] → SHIP else REGEN.
    Same arithmetic as paper_explore/glue/gate.py:transformer_seq_gate.

    Arch is dispatched from ckpt["hparams"]["arch"]; see
    `_build_head_module`. The same env-var path serves both attn_pool
    (Phase 4) and transformer_L2 (SYS25c).
    """

    def __init__(self, ckpt_path: str, tau_table_path: str) -> None:
        import json as _json
        ckpt = torch.load(ckpt_path, weights_only=False,
                          map_location="cpu")
        hp = ckpt["hparams"]
        self.arch = hp.get("arch")
        self.model = _build_head_module(hp)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        with open(tau_table_path) as _f:
            self.tau_table = _json.load(_f)
        self.global_tau = float(
            (self.tau_table.get("global") or {}).get("tau", 0.5)
        )
        self.per_source = self.tau_table.get("per_source", {}) or {}
        # Warmup so first inference doesn't pay PyTorch dispatch JIT cost
        with torch.inference_mode():
            _ = self.model(
                torch.zeros(1, 8, int(hp.get("n_features", 4))),
                torch.tensor([8]),
            )

    def tau_for(self, source: str | None) -> float:
        if source and source in self.per_source:
            return float(
                self.per_source[source].get("best", {}).get(
                    "tau", self.global_tau,
                )
            )
        return self.global_tau

    def decide(
        self,
        feature_seq: list[list[float]] | None,
        source: str | None,
    ) -> dict[str, Any] | None:
        if not feature_seq:
            return {"verdict": "REGEN", "score": None,
                    "tau": self.global_tau, "reason": "no_features"}
        # SYS55 cell C: head-forward STUB. Skip the tensor construction +
        # the 67k-param transformer matmul; return a constant score. The
        # feature pipeline (GPU per-step sync + accumulation) and the gate
        # plumbing (decide() call site, CompletionOutput.head_decision) are
        # left fully intact, so (B - C) isolates the head-forward cost and
        # (C - A) isolates the feature-pipeline cost. Decision-correctness
        # is irrelevant for this throughput-only cell.
        score = 0.5
        tau = self.tau_for(source)
        return {
            "verdict": "SHIP" if score >= tau else "REGEN",
            "score": score,
            "tau": tau,
            "source": source,
        }


# Back-compat alias: existing callers (output_processor.py, the
# paper_explore bench harness) import / log this name from Phase 4.
# The class is now arch-dispatched, so the name no longer literally
# means "attn_pool only" — it's the cascade head, arch read from
# the ckpt. Renaming the env vars would break the bench wiring.
InEngineAttnPoolHead = InEngineCascadeHead


# Singleton — loaded on first access, reused for all requests in the
# engine's lifetime. Env vars (unchanged from Phase 4 for backwards
# compat with the existing bench harness wiring; arch is read from
# the ckpt):
#   VLLM_CASCADE_ATTN_POOL_CKPT  — path to head .pt (AttnPool OR
#                                  TransformerSeq state_dict + hparams)
#   VLLM_CASCADE_ATTN_POOL_TAU   — path to tau-table .json
_HEAD_SINGLETON: "InEngineCascadeHead | None" = None
_HEAD_INIT_TRIED: bool = False


def get_in_engine_head() -> "InEngineCascadeHead | None":
    global _HEAD_SINGLETON, _HEAD_INIT_TRIED
    if _HEAD_SINGLETON is not None or _HEAD_INIT_TRIED:
        return _HEAD_SINGLETON
    _HEAD_INIT_TRIED = True
    import os as _os
    import logging as _logging
    ckpt = _os.environ.get("VLLM_CASCADE_ATTN_POOL_CKPT")
    tau = _os.environ.get("VLLM_CASCADE_ATTN_POOL_TAU")
    if not ckpt or not tau:
        return None
    try:
        _HEAD_SINGLETON = InEngineCascadeHead(ckpt, tau)
        _logging.getLogger(__name__).info(
            "[SYS25 Phase 4/c] Loaded in-engine cascade head arch=%s "
            "from ckpt=%s tau=%s",
            _HEAD_SINGLETON.arch, ckpt, tau,
        )
    except Exception as e:  # noqa: BLE001
        _logging.getLogger(__name__).warning(
            "[SYS25 Phase 4/c] Failed to load in-engine head: %s", e,
        )
        _HEAD_SINGLETON = None
    return _HEAD_SINGLETON
