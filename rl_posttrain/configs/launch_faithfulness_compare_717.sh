#!/usr/bin/env bash
set -euo pipefail

config="rl_posttrain/configs/faithfulness_compare_717.yaml"

python3 lilypad/launch.py "$config"

python3 lilypad/launch.py "$config" \
  -n alpamayo-rl-faithfulness-global717 \
  -o workload_variant_config.entrypoint_fn_config.workspace_dir /mnt/work/tmp/alpamayo_rl_faithfulness_global717 \
  -o workload_variant_config.entrypoint_fn_config.ckpt_s3_prefix alpamayo_rl/checkpoints/faithfulness_global717 \
  -o workload_variant_config.entrypoint_fn_config.wandb_experiment faithfulness_global717 \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DISABLE_CLIPGEN 1 \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID faithfulness-global717

python3 lilypad/launch.py "$config" \
  -n alpamayo-rl-faithfulness-judge717 \
  -o workload_variant_config.entrypoint_fn_config.workspace_dir /mnt/work/tmp/alpamayo_rl_faithfulness_judge717 \
  -o workload_variant_config.entrypoint_fn_config.reward_mode llm_judge \
  -o workload_variant_config.entrypoint_fn_config.ckpt_s3_prefix alpamayo_rl/checkpoints/faithfulness_judge717 \
  -o workload_variant_config.entrypoint_fn_config.wandb_experiment faithfulness_judge717 \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID faithfulness-judge717
