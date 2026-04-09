#!/usr/bin/env bash
set -euxo pipefail

mkdir -p /root/autodl-tmp/.pixi-home/bin \
  /root/autodl-tmp/.pixi-cache \
  /root/autodl-tmp/.cache/huggingface \
  /root/autodl-tmp/.cache/pip \
  /root/autodl-tmp/.cache/torch \
  /root/autodl-tmp/.cache/triton \
  /root/autodl-tmp/.cache/wandb \
  /root/autodl-tmp/rl/scripts \
  /root/autodl-tmp/rl/docs

if ! grep -q '### RL pixi env ###' ~/.bashrc; then
  cat >> ~/.bashrc <<'EOF'
### RL pixi env ###
export PIXI_HOME=/root/autodl-tmp/.pixi-home
export PIXI_CACHE_DIR=/root/autodl-tmp/.pixi-cache
export PATH=/root/autodl-tmp/.pixi-home/bin:$PATH
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/.cache/triton
export WANDB_DIR=/root/autodl-tmp/.cache/wandb
EOF
fi

export PIXI_HOME=/root/autodl-tmp/.pixi-home
export PIXI_CACHE_DIR=/root/autodl-tmp/.pixi-cache
export PATH=/root/autodl-tmp/.pixi-home/bin:$PATH

if [ ! -x /root/autodl-tmp/.pixi-home/bin/pixi ]; then
  curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/root/autodl-tmp/.pixi-home PIXI_BIN_DIR=/root/autodl-tmp/.pixi-home/bin PIXI_NO_PATH_UPDATE=1 bash
fi

pixi --version
cd /root/autodl-tmp/rl
sed -n '1,220p' pixi.toml
