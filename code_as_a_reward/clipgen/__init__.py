# SPDX-License-Identifier: Apache-2.0
"""Per-clip LLM-generated reward functions (VLM-CaR adapted, arXiv 2402.04764).

Pipeline: dossier.py (scene ground truth -> compact prompt text) ->
generate.py (claude-opus-5 writes a scene-specific reward function) ->
gate.py (empirical accept/reject on positive/negative trajectories) ->
run_prototype.py (5-clip end-to-end harness + report).

Design doc: https://claude.ai/code/artifact/14a64f93-9304-4037-97bb-69e1890ddf5b
"""

# This