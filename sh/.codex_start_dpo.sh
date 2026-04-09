#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/autodl-tmp/rl/logs
pkill -f 'pixi install -e dpo' || true
pkill -f 'pixi install -e grpo' || true
sleep 2
source /etc/network_turbo
export PIXI_HOME=/root/autodl-tmp/.pixi-home
export PIXI_CACHE_DIR=/root/autodl-tmp/.pixi-cache
export PATH=/root/autodl-tmp/.pixi-home/bin:$PATH
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/.cache/triton
export WANDB_DIR=/root/autodl-tmp/.cache/wandb
cd /root/autodl-tmp/rl
nohup bash -lc 'source /etc/network_turbo && export PIXI_HOME=/root/autodl-tmp/.pixi-home PIXI_CACHE_DIR=/root/autodl-tmp/.pixi-cache PATH=/root/autodl-tmp/.pixi-home/bin:$PATH HF_HOME=/root/autodl-tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip TORCH_HOME=/root/autodl-tmp/.cache/torch TRITON_CACHE_DIR=/root/autodl-tmp/.cache/triton WANDB_DIR=/root/autodl-tmp/.cache/wandb && cd /root/autodl-tmp/rl && pixi install -e dpo' > /root/autodl-tmp/rl/logs/pixi_dpo.log 2>&1 &
echo $! > /root/autodl-tmp/rl/logs/pixi_dpo.pid
cat /root/autodl-tmp/rl/logs/pixi_dpo.pid
sleep 5
head -n 40 /root/autodl-tmp/rl/logs/pixi_dpo.log || true
