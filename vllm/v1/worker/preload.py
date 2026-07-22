"""paper_explore SYS83 — Ray worker-process setup hook.

Fully imports the vllm package and the GPU worker module at Ray worker
start, BEFORE actor-class deserialization runs. In the ray-executor TP
actor path (ray.serve.llm), `from vllm import SamplingParams`-style
package-level re-exports intermittently fail with "cannot import name
... from 'vllm' (unknown location)" — a partially-initialized-package
observation during deserialization-time imports. Importing everything
eagerly here sidesteps the race for every re-export site at once.

Use via runtime_env:
    {"worker_process_setup_hook": "vllm.v1.worker.preload.preload_worker_modules"}
"""


def preload_worker_modules() -> None:
    import vllm  # noqa: F401
    import vllm.v1.worker.gpu_worker  # noqa: F401
