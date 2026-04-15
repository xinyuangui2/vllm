# SPDX-License-Identifier: Apache-2.0
"""Unified scheduling service for VLM serving.

Implements two complementary models (currently FIFO placeholders):
- Workload model (Σ m_i r_i): GPU load balancing / routing
- Cost model (Σ w_i C_i): Request batching with WSPT priority w_i/r_i

Reference: vlm_scheduling_design.pdf, Sections 3-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


@dataclass
class RequestEstimate:
    """Per-request workload and cost estimates."""

    request_id: str
    # Resource usage: m_i (Eq.8)
    #   m_i = ceil((L_i + O_bar) / block_size) + m_act_i
    resource_usage: float = 0.0
    # Remaining time: r_i (Eq.7)
    #   r_i = T_enc(P_i)*1[not encoded]
    #       + ceil(L_rem / C) * T_base
    #       + (O_bar - generated)+ * T_base
    remaining_time: float = 0.0
    # Weight: w_i = 1 + alpha * wait_time
    weight: float = 1.0
    # WSPT priority: w_i / r_i (Eq.3)
    priority: float = 0.0


@dataclass
class EMAState:
    """Exponential moving average tracker for online estimation."""

    value: float = 0.0
    count: int = 0
    beta: float = 0.1

    def update(self, observed: float) -> None:
        if self.count == 0:
            self.value = observed
        else:
            self.value = self.beta * observed + (1.0 - self.beta) * self.value
        self.count += 1


class VLMSchedulingService:
    """Unified scheduling service for VLM serving.

    Currently a FIFO placeholder. All ranking methods return requests
    in their original (arrival) order. The interface is designed for
    future WSPT-based scheduling.

    Two models (Section 3 of design doc):
        Workload (Σ m_i r_i): GPU routing — balance resource-time across GPUs.
        Cost (Σ w_i C_i): Batching — minimize weighted completion time.
            Optimal priority: w_i / r_i (WSPT, proven via interchange argument).
            Tiebreak: ascending m_i (less resource usage first).
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        # EMA trackers (Section 4)
        self.ema: dict[str, EMAState] = {
            "T_base": EMAState(),  # iteration time (Eq.5)
        }
        # Per-category output length EMA: O_bar_type (Eq.6)
        self.output_length_ema: dict[str, EMAState] = {}

    # ------------------------------------------------------------------
    # Workload model — GPU routing (Section 3.1)
    # ------------------------------------------------------------------

    def compute_workload(self, request: Request) -> float:
        """Compute m_i * r_i for a single request (Eq.1).

        Used for GPU routing: route to argmin_g (Workload_g + m_j * r_j^g).
        Placeholder: returns 0.
        """
        return 0.0

    def route_to_gpu(
        self, request: Request, gpu_workloads: list[float]
    ) -> int:
        """Select GPU for a new request (Section 5.4).

        GPU* = argmin_g (Workload_g + m_j * r_j^g).
        Placeholder: returns 0 (always first GPU).
        """
        return 0

    # ------------------------------------------------------------------
    # Cost model — request batching (Section 3.2)
    # ------------------------------------------------------------------

    def compute_priority(self, request: Request) -> float:
        """Compute WSPT priority w_i / r_i (Eq.3).

        Higher priority = should be admitted sooner.
        Placeholder: returns 0.
        """
        return 0.0

    def rank_pending(self, requests: list[Request]) -> list[Request]:
        """Rank pending requests by priority (Section 5.1).

        Target: sort by w_i/r_i descending, tiebreak by ascending m_i.
        Current: FIFO (returns input order unchanged).
        """
        return list(requests)

    # ------------------------------------------------------------------
    # Estimation helpers (Section 4)
    # ------------------------------------------------------------------

    def estimate_remaining_time(self, request: Request) -> float:
        """Estimate r_i for a request (Eq.7).

        r_i = T_enc(P_i) * 1[VLM, not encoded]     (known)
            + ceil(L_rem / C) * T_base               (known)
            + (O_bar_type - generated_i)+ * T_base    (estimated)

        Placeholder: returns 0.
        """
        return 0.0

    def estimate_resource_usage(self, request: Request) -> float:
        """Estimate m_i for a request (Eq.8).

        m_i = ceil((L_i + O_bar_type) / block_size)  (estimated lifetime KV)
            + m_act_i                                  (known activation memory)

        Placeholder: returns 0.
        """
        return 0.0

    def estimate_request(self, request: Request) -> RequestEstimate:
        """Compute full estimate for a request."""
        return RequestEstimate(
            request_id=request.request_id,
            resource_usage=self.estimate_resource_usage(request),
            remaining_time=self.estimate_remaining_time(request),
            priority=self.compute_priority(request),
        )

    # ------------------------------------------------------------------
    # Online EMA updates (Section 4.3-4.4)
    # ------------------------------------------------------------------

    def update_iteration_time(self, measured: float) -> None:
        """Update T_base EMA after each iteration (Eq.5)."""
        self.ema["T_base"].update(measured)

    def update_output_length(self, category: str, observed: int) -> None:
        """Update O_bar per category after request completes (Eq.6)."""
        if category not in self.output_length_ema:
            self.output_length_ema[category] = EMAState()
        self.output_length_ema[category].update(float(observed))

    def update_encoder_time(self, patch_bucket: int, measured: float) -> None:
        """Update T_enc^obs per patch-count bucket (Eq.9)."""
        key = f"T_enc_{patch_bucket}"
        if key not in self.ema:
            self.ema[key] = EMAState()
        self.ema[key].update(measured)

    # ------------------------------------------------------------------
    # Preemption (Section 5.2)
    # ------------------------------------------------------------------

    def should_preempt(
        self,
        new_request: Request,
        victim_candidate: Request,
    ) -> bool:
        """Check if new_request should preempt victim (Section 5.2).

        Evict when w_j * r_k > w_k * R_redo_k.
        Placeholder: returns False (never preempt).
        """
        return False
