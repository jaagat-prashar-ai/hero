# SPDX-License-Identifier: Apache-2.0
"""Rich per-clip/attempt W&B logging for the real-rollout gate loop
(run_prototype.py), so the fix for the GT-overfit bug (see run_prototype.py's
module docstring) is directly auditable: for every (clip, attempt), logs the
whole real rollout group overlaid on the scene's camera frame colored by
that attempt's candidate score (a "heatmap" of how the 12 rollouts vary and
score), a table with each rollout's reasoning text/score/component
breakdown, the candidate reward function source, and the gate verdict.

This is a standalone W&B run (this job never touches cosmos-rl/the training
controller), unlike rl_posttrain/rewards/code_reward_entry.py's
_get_overlay_run sibling-run pattern -- there is no controller run to be a
sibling of here.
"""

from __future__ import annotations

import html
import json
from typing import Any


class ClipgenWandbLogger:
    def __init__(self, project: str, entity: str | None = None, run_name: str | None = None):
        import wandb

        self._wandb = wandb
        self.run = wandb.init(project=project, entity=entity, name=run_name, job_type="clipgen-real-rollout-gate")
        self._frame_cache: dict[str, tuple[Any, dict, dict] | None] = {}
        self._chunk_index = None
        self._s3_client = None

    def _fetch_frame(self, clip_id: str):
        """Best-effort t0 camera frame + calibration for `clip_id` (cached
        per clip across attempts). None if the clip is outside the warm
        cache or any fetch step fails -- same best-effort convention as
        analyze_group_rollouts.render_overlays."""
        if clip_id in self._frame_cache:
            return self._frame_cache[clip_id]
        try:
            import boto3

            from code_as_a_reward.clipgen import build_overlays as bo

            if self._s3_client is None:
                self._s3_client = boto3.client("s3")
                self._chunk_index = bo._read_parquet_s3(self._s3_client, f"{bo.WARM_CACHE}/clip_index.parquet")
            result = bo.fetch_t0_frame(self._s3_client, clip_id, self._chunk_index, {})
        except Exception as e:
            print(f"[clipgen-wandb] {clip_id}: camera frame unavailable ({type(e).__name__}: {e})", flush=True)
            result = None
        self._frame_cache[clip_id] = result
        return result

    def log_attempt(
        self,
        clip_id: str,
        attempt: int,
        rollouts: list[dict[str, Any]],  # analyze_group_rollouts.select_and_verify's .scored
        argmax_rollout_id: int | None,
        gate_result,  # gate.GateResult | None
        source: str,
    ) -> None:
        wandb = self._wandb
        table = wandb.Table(columns=["rollout_id", "is_argmax", "score", "components", "coc_text"])
        for r in rollouts:
            table.add_data(
                r["rollout_id"],
                r["rollout_id"] == argmax_rollout_id,
                r.get("clipgen_score"),
                json.dumps(r.get("clipgen_components") or {}),
                r.get("coc_text", ""),
            )

        log: dict[str, Any] = {
            "clip_id": clip_id,
            "attempt": attempt,
            "argmax_rollout_id": argmax_rollout_id,
            "rollouts": table,
            "reward_fn_source": wandb.Html(f"<pre>{html.escape(source)}</pre>"),
            "passed": bool(gate_result.passed) if gate_result is not None else False,
            "pos_score": gate_result.pos_score if gate_result is not None else None,
            "max_pert": gate_result.max_pert if gate_result is not None else None,
        }
        if gate_result is not None and not gate_result.passed:
            log["gate_feedback"] = gate_result.feedback()

        fetched = self._fetch_frame(clip_id)
        if fetched is not None:
            frame, cam_intr, cam_extr = fetched
            try:
                from code_as_a_reward.clipgen.analyze_group_rollouts import render_multi_overlay

                overlay = render_multi_overlay(frame.copy(), rollouts, cam_intr, cam_extr, argmax_rollout_id)
                log["overlay"] = wandb.Image(
                    overlay,
                    caption=f"{clip_id} attempt {attempt}: {len(rollouts)} rollouts, argmax={argmax_rollout_id}",
                )
            except Exception as e:
                print(f"[clipgen-wandb] {clip_id}: overlay render failed ({type(e).__name__}: {e})", flush=True)

        self.run.log(log)

    def finish(self) -> None:
        self.run.finish()
