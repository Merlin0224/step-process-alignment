#!/usr/bin/env bash
set -u

MODEL_DIR=/root/autodl-tmp/models/Qwen/Qwen2.5-3B
SHARD1="$MODEL_DIR/model-00001-of-00002.safetensors"
SHARD1_PART="$MODEL_DIR/model-00001-of-00002.safetensors.part"
SHARD2="$MODEL_DIR/model-00002-of-00002.safetensors"
DOWNLOAD_SCRIPT=/root/autodl-tmp/rl/logs/resume_qwen_http.py
DOWNLOAD_LOG=/root/autodl-tmp/rl/logs/resume_qwen_http.log
GRPO_PY=/root/autodl-tmp/rl/envs/grpo/.pixi/envs/grpo/bin/python
VERL_ROOT=/root/autodl-tmp/rl/external/verl
STATUS_LOG=/root/autodl-tmp/rl/logs/qwen_verl_watch.log

mkdir -p /root/autodl-tmp/rl/logs
mkdir -p /root/autodl-tmp/rl/external

log() {
  printf '[%s] %s
' "$(date '+%F %T')" "$*" | tee -a "$STATUS_LOG"
}

restart_downloader() {
  log "restarting qwen downloader screen"
  screen -S qwen_download -X quit >/dev/null 2>&1 || true
  screen -dmS qwen_download bash -lc "source /etc/network_turbo >/dev/null 2>&1 || true; exec $GRPO_PY -u $DOWNLOAD_SCRIPT >> $DOWNLOAD_LOG 2>&1"
}

install_verl() {
  log "starting veRL installation"
  source /etc/network_turbo >/dev/null 2>&1 || true

  if ! "$GRPO_PY" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("pip") else 1)
PY
  then
    log "bootstrapping pip with ensurepip"
    "$GRPO_PY" -m ensurepip --upgrade >> "$STATUS_LOG" 2>&1
  fi

  if [ ! -d "$VERL_ROOT/.git" ]; then
    log "cloning verl repository"
    git clone https://github.com/volcengine/verl.git "$VERL_ROOT" >> "$STATUS_LOG" 2>&1
  else
    log "updating verl repository"
    git -C "$VERL_ROOT" fetch --all --prune >> "$STATUS_LOG" 2>&1 || true
    git -C "$VERL_ROOT" pull --ff-only >> "$STATUS_LOG" 2>&1 || true
  fi

  log "installing verl with --no-deps -e ."
  (cd "$VERL_ROOT" && "$GRPO_PY" -m pip install --no-deps -e .) >> "$STATUS_LOG" 2>&1

  log "verifying verl import"
  "$GRPO_PY" - <<'PY' >> "$STATUS_LOG" 2>&1
import verl
print('verl_import_ok', getattr(verl, '__file__', 'unknown'))
PY
  log "veRL installation completed"
}

log "watcher started"
while true; do
  if [ -f "$SHARD1" ] && [ -f "$SHARD2" ] && [ ! -f "$SHARD1_PART" ]; then
    log "model download complete"
    if "$GRPO_PY" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("verl") else 1)
PY
    then
      log "verl already installed"
    else
      install_verl
    fi
    exit 0
  fi

  if [ -f "$SHARD1_PART" ]; then
    part_size=$(du -sh "$SHARD1_PART" 2>/dev/null | awk '{print $1}')
    log "model still downloading: shard1.part=$part_size"
  else
    log "model not complete yet and part file missing"
  fi

  if ! screen -ls 2>/dev/null | grep -q 'qwen_download'; then
    log "qwen_download screen missing before completion"
    restart_downloader
  fi

  sleep 300
done
