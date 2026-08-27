#!/usr/bin/env bash
set -euo pipefail

config="rl_posttrain/configs/faithfulness_compare_fivehour75.yaml"
suffix="${FIVEHOUR_CANARY_SUFFIX:-v4}"

python3 lilypad/launch.py "$config" \
  -n "alpamayo-rl-faithfulness-fivehour75-clipgen-${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.workspace_dir "/mnt/work/tmp/alpamayo_rl_faithfulness_fivehour75_clipgen_${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.ckpt_s3_prefix "alpamayo_rl/checkpoints/faithfulness_fivehour75_clipgen_${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.wandb_experiment "faithfulness_fivehour75_clipgen_${suffix}" \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID "faithfulness-fivehour75-clipgen-${suffix}"

python3 lilypad/launch.py "$config" \
  -n "alpamayo-rl-faithfulness-fivehour75-global-${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.workspace_dir "/mnt/work/tmp/alpamayo_rl_faithfulness_fivehour75_global_${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.ckpt_s3_prefix "alpamayo_rl/checkpoints/faithfulness_fivehour75_global_${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.wandb_experiment "faithfulness_fivehour75_global_${suffix}" \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DISABLE_CLIPGEN enabled \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID "faithfulness-fivehour75-global-${suffix}"

python3 lilypad/launch.py "$config" \
  -n "alpamayo-rl-faithfulness-fivehour75-judge-${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.workspace_dir "/mnt/work/tmp/alpamayo_rl_faithfulness_fivehour75_judge_${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.reward_mode llm_judge \
  -o workload_variant_config.entrypoint_fn_config.ckpt_s3_prefix "alpamayo_rl/checkpoints/faithfulness_fivehour75_judge_${suffix}" \
  -o workload_variant_config.entrypoint_fn_config.wandb_experiment "faithfulness_fivehour75_judge_${suffix}" \
  -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID "faithfulness-fivehour75-judge-${suffix}"
