#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/autodl-tmp/rl/envs/grpo/.pixi /root/autodl-tmp/rl/logs
pkill -f 'pixi install -e dpo' || true
pkill -f 'pixi install -e grpo' || true
sleep 2
cd /root/autodl-tmp/rl/envs/grpo
rm -f pixi.lock
nohup bash -lc 'source /etc/network_turbo && export PIXI_HOME=/root/autodl-tmp/.pixi-home PIXI_CACHE_DIR=/root/autodl-tmp/.pixi-cache PATH=/root/autodl-tmp/.pixi-home/bin:$PATH HF_HOME=/root/autodl-tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip TORCH_HOME=/root/autodl-tmp/.cache/torch TRITON_CACHE_DIR=/root/autodl-tmp/.cache/triton WANDB_DIR=/root/autodl-tmp/.cache/wandb && cd /root/autodl-tmp/rl/envs/grpo && pixi -vv install -e grpo' > /root/autodl-tmp/rl/logs/pixi_grpo_isolated.log 2>&1 &
echo $! > /root/autodl-tmp/rl/logs/pixi_grpo_isolated.pid
cat /root/autodl-tmp/rl/logs/pixi_grpo_isolated.pid
sleep 8
tail -n 80 /root/autodl-tmp/rl/logs/pixi_grpo_isolated.log || true
