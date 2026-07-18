#!/usr/bin/env bash
set -euo pipefail

remote="${PETRIDISH_LAB_REMOTE:-m@192.168.0.203}"
deploy_root="${PETRIDISH_LAB_DEPLOY_ROOT:-/home/m/Code/petridish-lab-deploy2}"
service="${PETRIDISH_LAB_SERVICE:-petridish-lab-v12.service}"
branch="${PETRIDISH_LAB_BRANCH:-codex/experiment-lab}"
mode="${1:-full}"

if [[ "$mode" != "full" && "$mode" != "frontend-only" ]]; then
  echo "usage: $0 [full|frontend-only]" >&2
  exit 64
fi

(
  cd frontend
  npm run check
  npm run build
)

if [[ "$mode" == "frontend-only" ]]; then
  scp -r frontend/dist/. "$remote:$deploy_root/frontend/dist/"
  echo "frontend deployed without restarting the laboratory supervisor"
  exit 0
fi

trainer_pattern="^$deploy_root/.venv/bin/python -m petridish.train_shakespeare"
if ssh "$remote" "pgrep -f '$trainer_pattern' >/dev/null"; then
  echo "refusing full deploy: a persistent organism trainer is active" >&2
  echo "use '$0 frontend-only' or wait for checkpointed trainers to stop" >&2
  exit 2
fi

ssh "$remote" "set -e; cd '$deploy_root'; git fetch origin '$branch'; git checkout --detach FETCH_HEAD"
scp -r frontend/dist/. "$remote:$deploy_root/frontend/dist/"
ssh "$remote" "systemctl --user restart '$service'; systemctl --user is-active '$service'"

