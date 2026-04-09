#!/usr/bin/env bash
source /etc/network_turbo >/dev/null 2>&1 || true
exec /root/autodl-tmp/rl/envs/grpo/.pixi/envs/grpo/bin/python -u /root/autodl-tmp/rl/logs/resume_qwen_http.py
