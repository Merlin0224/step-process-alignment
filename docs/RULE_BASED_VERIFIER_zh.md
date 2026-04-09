# Rule-Based Verifier 使用说明

这版 `Verifier` 先面向 `GSM8K`，目标是给阶段 1 和后续 `MCTS / Step-DPO / Step-GRPO` 提供一个稳定的规则校验入口。

文件位置：

- `scripts/rule_based_verifier.py`

## 当前规则

1. 优先从模型输出中抽取 `<answer>...</answer>`
2. 优先从 `GSM8K` 标注答案中的 `#### final_answer` 抽取标准答案
3. 如果双方都能解析成数值，则做严格数值比较
4. 如果数值解析失败，则退回轻量文本比较

支持的答案形式包括：

- 整数，例如 `72`
- 带千分位的整数，例如 `1,234`
- 小数，例如 `0.5`
- 分数，例如 `1/2`
- 百分数，例如 `12.5%`

## 单条样例

```bash
python scripts/rule_based_verifier.py \
  --prediction "Let's solve it. <answer>72</answer>" \
  --reference "Some reasoning here. #### 72"
```

如果预测正确，脚本会输出一条 JSON，并返回退出码 `0`。

## 批量评测 JSONL

默认约定 JSONL 每一行至少包含两个字段：

- `prediction`
- `answer`

示例：

```json
{"prediction": "...\n<answer>72</answer>", "answer": "The result is 72. #### 72"}
{"prediction": "...\n<answer>18</answer>", "answer": "The result is 20. #### 20"}
```

批量运行：

```bash
python scripts/rule_based_verifier.py --jsonl path/to/preds.jsonl
```

如果字段名不同，可以显式指定：

```bash
python scripts/rule_based_verifier.py \
  --jsonl path/to/preds.jsonl \
  --prediction-field response \
  --reference-field gold_answer
```

## 严格模式与回退模式

默认是严格模式：

- 如果模型没有输出 `<answer>...</answer>`，会直接判错

如果你在调试早期模型，想让脚本在缺标签时回退到“抽最后一个数”，可以加：

```bash
python scripts/rule_based_verifier.py \
  --prediction "The final answer is 72" \
  --reference "Reasoning #### 72" \
  --allow-fallback
```

## 自检

```bash
python scripts/rule_based_verifier.py --self-check
```

这会跑一组内置样例，确认以下逻辑正常：

- 标签抽取
- `####` 标准答案抽取
- 千分位归一化
- 分数与小数等价比较
- 严格模式下缺标签直接判错

## 下一步建议

写完这个 `Verifier` 后，可以直接接下面两步：

1. 从 `GSM8K train` 抽 `1000-2000` 题形成核心实验集
2. 统一模型输出格式为 `<answer>...</answer>`，把这个脚本接进 MCTS rollout 的自动打分流程
