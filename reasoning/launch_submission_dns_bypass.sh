#!/usr/bin/env bash
set -euo pipefail

: "${AUTH_TOKEN:?export AUTH_TOKEN first}"
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"
: "${HF_TOKEN:?export HF_TOKEN first}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hosts_file="$(mktemp)"
trap 'rm -f "${hosts_file}"' EXIT
cp /etc/hosts "${hosts_file}"
printf '\n137.131.42.171 ml-infra.applied.dev\n' >> "${hosts_file}"

cd "${repo_root}"
unshare -Ur -m bash -c '
  mount --bind "$1" /etc/hosts
  shift
  export GRPC_DNS_RESOLVER=native
  exec "$@"
' bash "${hosts_file}" \
  python3 lilypad/launch.py reasoning/configs/code_consistency_submission.yaml \
  --skip-dependency-validation \
  -n alpamayo-code-consistency-track1-submit-r1
