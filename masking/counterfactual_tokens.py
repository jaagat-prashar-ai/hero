# SPDX-License-Identifier: Apache-2.0
"""
counterfactual_tokens.py — Logit-based counterfactual token analysis for CoT reasoning.

Two analyses:
  token_alternative_map(data)   — pure logit inspection, no re-running.
  counterfactual_sweep(data)    — forces top-K alternatives via LogitsProcessor,
                                   re-samples subsequent reasoning, compares trajectories.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from masking.masked_model import MaskedAlpamayo1_5

logger = logging.getLogger(__name__)


class ForcedTokenAtStep(LogitsProcessor):
    """Zero-out all logits except `token_id` at generation step `step`.

    Step is 0-indexed relative to the first NEW token (not prompt tokens).
    Append LAST in LogitsProcessorList so ExpertLogitsProcessor runs first.
    """

    def __init__(self, step: int, token_id: int) -> None:
        self.step = step
        self.token_id = token_id
        self._counter = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self._counter == self.step:
            scores[:] = torch.finfo(scores.dtype).min
            scores[:, self.token_id] = 0.0
        self._counter += 1
        return scores


# --------------------------------------------------------------------------- #
# Output data classes                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class AlternativeToken:
    token_id: int
    text: str
    prob: float   # softmax probability in the original (unforced) distribution


@dataclass
class ReasoningPosition:
    """Logit statistics for one generated reasoning token."""

    step: int            # 0-indexed generation step (= col - prompt_len)
    col: int             # absolute column in vlm_outputs.sequences
    sampled_id: int
    sampled_text: str
    sampled_prob: float  # rank-0 probability in the original distribution
    entropy: float       # Shannon entropy in nats
    top_k: list[AlternativeToken]  # sorted by prob descending; rank-0 = sampled token


@dataclass
class CounterfactualResult:
    """Trajectory delta from forcing one alternative token at one reasoning step."""

    forced_token: AlternativeToken
    forced_cot: str          # full CoT text when this token was forced
    d_curvature_mean: float  # mean |Δcurvature| over waypoints vs. baseline
    d_curvature_max: float
    endpoint_shift_m: float  # L2 distance of final waypoint vs. baseline (metres)
    traj_ade_m: float        # average displacement error over full trajectory (metres)
