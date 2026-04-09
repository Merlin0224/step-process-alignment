# GSM8K 核心实验集抽样与预处理

对应脚本：

- `scripts/prepare_gsm8k_core_set.py`

这份脚本默认面向服务器环境，不需要在本地下载数据集。推荐直接在服务器上对已经 `save_to_disk` 的 `GSM8K` 目录运行。

## 默认输入/输出

默认输入：

- `/root/autodl-tmp/datasets/gsm8k/main`

默认输出：

- `/root/autodl-tmp/datasets/gsm8k/core`

输出内容包括：

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `summary.json`

## 默认切分策略

- `train-size = 1500`
- `dev-size = 200`
- `test-size = 0`

其中：

- `train` 和 `dev` 都从官方 `train` split 中按固定随机种子抽样
- `test-size <= 0` 时，保留完整官方 `test`

## 产出字段

每条样本会整理为统一结构，核心字段有：

- `problem_id`
- `question`
- `solution`
- `answer`
- `final_answer`
- `final_answer_numeric`
- `answer_tag`
- `prompt`

其中：

- `solution` 是 `#### final_answer` 之前的推导文本
- `final_answer` 是从官方答案中抽出的最终答案
- `answer_tag` 是后续可直接拿来做监督格式对齐的 `<answer>...</answer>`
- `prompt` 是带有答案标签约束的基础推理提示词

## 服务器运行示例

```bash
cd /root/autodl-tmp/rl
pixi run -e grpo python scripts/prepare_gsm8k_core_set.py
```

如果你想显式指定路径：

```bash
pixi run -e grpo python scripts/prepare_gsm8k_core_set.py \
  --dataset-path /root/autodl-tmp/datasets/gsm8k/main \
  --output-dir /root/autodl-tmp/datasets/gsm8k/core
```

## 调整核心实验规模

例如抽 `1200` 条训练、`200` 条开发，并把测试集也裁成 `500` 条：

```bash
pixi run -e grpo python scripts/prepare_gsm8k_core_set.py \
  --train-size 1200 \
  --dev-size 200 \
  --test-size 500 \
  --seed 42
```

## 自定义 prompt 模板

如果你想把提示词改成自己的格式，可以准备一个模板文件，并确保其中包含 `{question}` 占位符：

```text
You are a careful solver.
Think step by step.
Return the final answer in <answer>...</answer>.

Problem:
{question}
```

然后运行：

```bash
pixi run -e grpo python scripts/prepare_gsm8k_core_set.py \
  --prompt-template-file /path/to/prompt_template.txt
```

## 本地轻量自检

这一步不依赖真实数据集，只检查脚本逻辑：

```bash
py scripts/prepare_gsm8k_core_set.py --self-check
```

## 下一步建议

这个脚本跑完后，通常可以直接接三件事：

1. 用 `train.jsonl` 做 MCTS 核心题池
2. 用 `dev.jsonl` 做参数调试和格式验收
3. 用 `test.jsonl` 做 `Pass@1` 离线评测
