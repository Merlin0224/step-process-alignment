# MCTS 数据生成与步级价值估计骨架

对应脚本：

- `scripts/build_mcts_value_data.py`

## 这版脚本做了什么

这不是最终版 UCT/PUCT 搜索器，而是阶段 2 的第一版可运行骨架。它先把最关键的链路打通：

1. 读取 `GSM8K core` 的 `train.jsonl`
2. 对每道题采样若干条初始推理路径
3. 把每条路径按“步”切成前缀状态
4. 对每个前缀做 Monte Carlo rollout
5. 用 `Rule-based Verifier` 估计步级价值 `Q_t`
6. 把树状数据保存成 JSONL

## 为什么先这样做

当前这版优先采用“前缀回放 + rollout 估值”的保守实现，而不是一步到位做完整树搜索，原因是：

- 更适合先排查 `vLLM / 显存 / prompt 拼接 / verifier` 这些工程问题
- 能直接产出后续 `Step-DPO` 需要的前缀价值数据
- 之后可以自然扩展出更完整的 selection / expansion / backup 逻辑

所以你可以把它理解成：

- 第一版 `Data & Value Engine`
- 可运行的 `MCTS skeleton`
- 面向后续扩展的树状数据采样器

## 输入与输出

默认输入：

- `/root/autodl-tmp/datasets/gsm8k/core/train.jsonl`

默认输出：

- `/root/autodl-tmp/datasets/gsm8k/mcts/train_trees.jsonl`
- `/root/autodl-tmp/datasets/gsm8k/mcts/train_trees_summary.json`

每条输出样本包含：

- `problem_id`
- `question`
- `prompt`
- `reference_answer`
- `nodes`
- `initial_paths`
- `statistics`

其中 `nodes` 里的关键字段有：

- `node_id`
- `parent_id`
- `depth`
- `step_text`
- `prefix_text`
- `q_value`
- `success_count`
- `rollout_count`
- `evaluation_mode`

## 默认运行方式

先用服务器环境实际跑少量样本：

```bash
cd /root/autodl-tmp/rl
pixi run -e grpo python scripts/build_mcts_value_data.py \
  --backend vllm \
  --model /root/autodl-tmp/models/Qwen/Qwen2.5-3B \
  --input-jsonl /root/autodl-tmp/datasets/gsm8k/core/train.jsonl \
  --output-jsonl /root/autodl-tmp/datasets/gsm8k/mcts/train_trees.jsonl \
  --max-problems 10 \
  --initial-branches 2 \
  --rollout-samples 3 \
  --max-step-depth 8 \
  --max-new-tokens 256 \
  --store-rollouts
```

## 适合 3090 的起步参数

建议先从下面这组开始：

- `max-problems = 10`
- `initial-branches = 2`
- `rollout-samples = 3`
- `max-step-depth = 6~8`
- `max-new-tokens = 192~256`
- `tensor-parallel-size = 1`
- `gpu-memory-utilization = 0.80~0.85`

等链路稳定后，再往上加：

- `max-problems`
- `initial-branches`
- `rollout-samples`

## 本地轻量自检

这个自检不依赖真实数据集，也不需要本地 vLLM：

```bash
py scripts/build_mcts_value_data.py --self-check
```

## 当前局限

这版脚本目前还有几处是刻意保守的：

1. 还没有完整实现 UCT / PUCT 的 selection 策略
2. 还没有做节点去重的语义归并，只按文本前缀去重
3. rollout 还是“从前缀继续生成”，没有做更复杂的 tree policy / backup policy
4. 还没有单独拆出 “tree policy / expansion / backup” 模块

这些都适合作为阶段 2 的下一轮迭代。

## 下一步建议

这份骨架同步到服务器并跑通后，最自然的下一步是：

1. 先对 `10-20` 道题跑一版小样本树数据
2. 抽查 JSONL 里的 `nodes.q_value` 是否符合直觉
3. 再把输出格式固定下来，用于后续 `Step-DPO` 偏好对构造
