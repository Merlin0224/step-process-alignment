#!/usr/bin/env bash
set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
cd /root/autodl-tmp/rl/envs/grpo
/root/autodl-tmp/.pixi-home/bin/pixi run -e grpo python /root/autodl-tmp/rl/scripts/build_mcts_value_data.py   --backend vllm   --model /root/autodl-tmp/models/Qwen/Qwen2.5-3B   --input-jsonl /root/autodl-tmp/datasets/gsm8k/core/train.jsonl   --output-jsonl /root/autodl-tmp/datasets/gsm8k/mcts/stage2_upgrade_500_trees.jsonl   --summary-json /root/autodl-tmp/datasets/gsm8k/mcts/stage2_upgrade_500_summary.json   --max-problems 500   --root-expansion-branches 2   --expansion-branches 2   --rollout-samples 3   --num-simulations 6   --max-step-depth 8   --step-max-new-tokens 96   --rollout-max-new-tokens 160   --gpu-memory-utilization 0.80   --max-model-len 2048   --save-every 1   --resume   --store-rollouts
