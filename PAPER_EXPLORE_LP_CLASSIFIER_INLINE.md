# paper_explore: inline per-request logprob aggregate stats

Branch `lp-classifier-inline` adds a thin patch to vLLM that lets a
request opt into per-request aggregate logprob statistics, computed
inline during decoding without serializing per-token logprobs to CPU.
The result lands in `CompletionOutput.aggregate_logprob_stats`.

Used by paper_explore SYS22 cascade-routing experiments. The downstream
"deciders" (`lp_only` MLP, transformer-on-per-token-sequence) consume
these 4 scalars to predict whether the cheap draft model's answer is
correct, and threshold a cascade ship/escalate decision.

## API

```python
sp = SamplingParams(
    max_tokens=128, temperature=0.0,
    logprobs=20,                          # required for top-K stats
    emit_aggregate_logprob_stats=True,    # opt in
)
# ... run engine ...
for completion in request_output.outputs:
    stats = completion.aggregate_logprob_stats
    # stats = {"mean_logprob": -0.42, "min_logprob": -1.31,
    #          "mean_max_prob": 0.78,  "neg_mean_entropy": -0.45}
```

The 4 scalars are exactly the features the `lp_only` head consumes
(plus a `pos_fraction=1.0` constant). Driver-side, just run the small
MLP on the 4 floats; with the inline patch you avoid serializing
per-token logprob dicts to CPU.

## What this patch adds

| file | change |
|---|---|
| `vllm/sampling_params.py` | new field `emit_aggregate_logprob_stats: bool = False` |
| `vllm/outputs.py` | new field `CompletionOutput.aggregate_logprob_stats: dict[str, float] \| None = None` |
| `vllm/v1/outputs.py` | per-step `EngineCoreOutput.aggregate_logprob_stats_delta` (running sums + step count + min) |
| `vllm/v1/engine/__init__.py` | schema for the per-request stats handoff |
| `vllm/v1/worker/gpu_input_batch.py` | per-request flag tracking |
| `vllm/v1/worker/gpu_model_runner.py` | **core change** — per-step accumulator (see below) |
| `vllm/v1/core/sched/scheduler.py` | pass-through |
| `vllm/v1/engine/output_processor.py` | pass-through + final finalize: compute means from accumulated sums when request finishes |

## Per-step accumulator (gpu_model_runner.py)

After the sampler computes the per-step top-K logprobs (which it
already does when `logprobs >= K` is set in SamplingParams), the
gpu_model_runner can compute the 4 per-step stats with no extra
forward passes:

```python
# Pseudocode at the point where sampler has chosen the next token
# and computed top-K logprobs for the step:
#   sampled_token_id:  scalar
#   topk_logprobs:     [K] tensor
#   topk_token_ids:    [K] tensor
# For each request opted in:
chosen_lp = topk_logprobs[idx_of_sampled_token]  # or recompute if not in top-K
max_prob = torch.exp(topk_logprobs[0])           # top-1 prob (top-K is sorted)
probs = torch.exp(topk_logprobs)
z = probs.sum()
neg_ent = (probs / z * (torch.log(probs / z))).sum()  # top-K entropy
per_request_state[req_id].update(chosen_lp, max_prob, neg_ent)
```

Per-request state is a small CUDA tensor (4 floats + 1 int per request)
allocated at engine init and kept on-device for the request's lifetime.

## Finalize (scheduler.py / output_processor.py)

When a request finishes (EOS / max_tokens / stop), the per-request state
is read out:

```python
state = per_request_state[req_id]
n = state.n_steps
if n > 0:
    completion.aggregate_logprob_stats = {
        "mean_logprob":     state.sum_chosen_lp / n,
        "min_logprob":      state.min_chosen_lp,
        "mean_max_prob":    state.sum_max_prob / n,
        "neg_mean_entropy": -state.sum_pos_entropy / n,
    }
```

## What this patch DOESN'T do (yet)

* **Inline MLP forward.** The driver still runs the small MLP on the 4
  scalars. Adding the MLP forward inline (so the cascade decision lands
  in the RequestOutput) is a follow-up: load the MLP from a
  configurable path (e.g. `VLLM_LP_CLASSIFIER_CKPT`) at engine init,
  add `CompletionOutput.cascade_score` + `cascade_decision`.

* **Suppress per-token logprob serialization.** When
  `emit_aggregate_logprob_stats=True` and `logprobs>=K`, vLLM still
  emits the per-token logprob dicts. Optimization: optionally skip
  per-token serialization if only the aggregate is needed. Saves a
  few ms per long-CoT request.

* **Streaming early-exit.** The aggregate is computed once at request
  finalization. Could be exposed every N steps so a streaming
  dispatcher can early-exit on confident requests. Larger change.

## Correctness validation plan

1. Build worker image from `lp-classifier-inline` branch.
2. Run a small extract with `emit_aggregate_logprob_stats=True`.
3. Compare the 4 stats to the same 4 stats computed driver-side from
   `outputs[0].logprobs` over the same generation (existing path).
4. They should match within float-precision tolerance.

If they match, switch all downstream paper_explore code paths to the
new field, drop the per-token-logprob serialization in the bench path
(saves ~5% latency on long-CoT mathvista), archive the legacy
cascade-prod-fixes branch.

## Status

* ✅ schema patches: `sampling_params.py` field, `outputs.py` field
* ✅ per-request accumulator dataclass + engine-level registry
   (`vllm/v1/cascade_lp_classifier.py`)
* ✅ **driver-side reference implementation wired through
   output_processor**: setting `emit_aggregate_logprob_stats=True`
   today populates `CompletionOutput.aggregate_logprob_stats` from
   the existing per-token `SampleLogprobs`. Same math as the inline
   accumulator — useful as a numerical reference and lets
   paper_explore exercise the API without waiting for the GPU-side
   integration.
* ⏳ TODO: per-step accumulator in gpu_model_runner (inline path —
   avoids serializing per-token logprob dicts to CPU)
* ⏳ TODO: build worker image, run paired extract, confirm inline
   accumulator matches the driver-side reference within fp32
   tolerance, then archive the legacy `cascade-prod-fixes` fork.

Estimated remaining engineering: ~1-2 days for the GPU-side
accumulator + numerical-equivalence test.
