#!/usr/bin/env bash
# Launch one AgencyBench arm per harness instance, all in parallel.
#
# Each arm gets its own `dsh web` server (so one slow session cannot block
# another), its own private agent root and grader root (see bench_agency.py),
# and its own log files under runs/.
#
#   run_arms.sh <spec> [<spec> ...]
#
# A spec is `<mode>:<preset>:<model-name>:<port>`, for example
# `v1:lh-v1:dsh-v1-r1:62110`.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$HERE/.." && pwd)"
HARNESS_ROOT="$(cd "$EVAL_ROOT/.." && pwd)/deepseek-harness"
SCENARIO="$EVAL_ROOT/bench/AgencyBench/AgencyBench-v2/Backend/scenario1"
RUNS="$EVAL_ROOT/runs"
export PATH="/Users/bytedance/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback:$PATH"
export DSH_PERMISSION_MODE=danger-full-access
export DEEPSEEK_API_KEY="$(grep -o 'sk-[A-Za-z0-9]*' "$HOME/.dsh/.credentials.yaml" | head -1)"

mkdir -p "$RUNS"
echo "[ARMS] starting at $(date -u +%FT%TZ)"

# 1. One harness per arm.
for spec in "$@"; do
  IFS=':' read -r mode preset model port <<<"$spec"
  (
    cd "$HARNESS_ROOT" || exit 1
    exec pnpm dsh web --port "$port"
  ) >"$RUNS/harness-$model.log" 2>&1 &
  echo "[ARMS] harness $model on port $port (pid $!)"
done

# 2. Wait until every harness answers before sending any prompt.
for spec in "$@"; do
  IFS=':' read -r mode preset model port <<<"$spec"
  for _ in $(seq 1 60); do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/"; then break; fi
    sleep 2
  done
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/" || true)"
  echo "[ARMS] harness $model http=$code"
done

# 3. One runner per arm.
for spec in "$@"; do
  IFS=':' read -r mode preset model port <<<"$spec"
  # `-u`: every [TURN]/[SCORE] line reaches the log as it happens, which is the
  # only progress signal a multi-hour arm offers.
  python3 -u "$HERE/bench_agency.py" \
    --base-url "http://127.0.0.1:$port" \
    --scenario "$SCENARIO" \
    --mode "$mode" \
    --preset "$preset" \
    --model-name "$model" \
    --max-subtasks 5 --second-pass \
    --turn-timeout 3000 --turn-retries 3 \
    --out "$RUNS/$model.json" \
    >"$RUNS/$model.log" 2>&1 &
  echo "[ARMS] runner $model (pid $!) -> $RUNS/$model.log"
done

wait
echo "[ARMS] all runners finished at $(date -u +%FT%TZ)"
