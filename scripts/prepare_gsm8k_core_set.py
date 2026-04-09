"""GSM8K 核心实验集抽样与预处理脚本。

设计目标：
1. 直接对接 `datasets.save_to_disk()` 保存的 GSM8K 目录
2. 从官方 train split 中稳定抽取核心实验训练集/开发集
3. 产出后续 Verifier / MCTS / Step-DPO / Step-GRPO 都能复用的 JSONL
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rule_based_verifier import extract_gsm8k_reference, parse_numeric_answer

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


DEFAULT_PROMPT_TEMPLATE = """You are a careful math reasoning assistant.
Solve the problem step by step.
Put the final answer inside <answer>...</answer>.

Question:
{question}
"""


@dataclass(frozen=True)
class PreparedExample:
    """统一后的样本结构。"""

    problem_id: str
    dataset_name: str
    dataset_config: str
    split: str
    source_index: int
    question: str
    solution: str
    answer: str
    final_answer: str
    final_answer_numeric: str | None
    answer_tag: str
    prompt: str


def normalize_solution_text(answer: str) -> tuple[str, str]:
    """把 GSM8K 原始 answer 拆成推导过程与最终答案。"""
    final_answer = extract_gsm8k_reference(answer)
    if final_answer is None:
        raise ValueError("未能从 GSM8K answer 字段中抽取 `#### final_answer`。")

    solution = answer.split("####", maxsplit=1)[0].rstrip()
    return solution, final_answer


def decimal_to_string(value: Decimal | None) -> str | None:
    """把 Decimal 转成更稳定的字符串。"""
    if value is None:
        return None
    normalized = value.normalize()
    as_text = format(normalized, "f")
    if "." in as_text:
        as_text = as_text.rstrip("0").rstrip(".")
    return as_text or "0"


def make_problem_id(split: str, source_index: int) -> str:
    """生成稳定问题 ID。"""
    return f"gsm8k-main-{split}-{source_index:05d}"


def prepare_record(
    record: dict[str, Any],
    *,
    split: str,
    source_index: int,
    prompt_template: str,
) -> PreparedExample:
    """把原始 GSM8K 样本整理成统一结构。"""

    question = str(record["question"]).strip()
    raw_answer = str(record["answer"]).strip()
    solution, final_answer = normalize_solution_text(raw_answer)
    final_answer_numeric = decimal_to_string(parse_numeric_answer(final_answer))

    return PreparedExample(
        problem_id=make_problem_id(split, source_index),
        dataset_name="gsm8k",
        dataset_config="main",
        split=split,
        source_index=source_index,
        question=question,
        solution=solution,
        answer=raw_answer,
        final_answer=final_answer,
        final_answer_numeric=final_answer_numeric,
        answer_tag=f"<answer>{final_answer}</answer>",
        prompt=prompt_template.format(question=question),
    )


def dataset_to_records(
    dataset: "Dataset",
    *,
    split: str,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """把 Hugging Face Dataset 转成可写 JSONL 的列表。"""

    prepared: list[dict[str, Any]] = []
    for source_index, record in enumerate(dataset):
        example = prepare_record(
            record,
            split=split,
            source_index=source_index,
            prompt_template=prompt_template,
        )
        prepared.append(asdict(example))
    return prepared


def sample_train_and_dev(
    train_dataset: "Dataset",
    *,
    train_size: int,
    dev_size: int,
    seed: int,
) -> tuple["Dataset", "Dataset"]:
    """从官方 train split 中稳定切出 train/dev。"""

    required = train_size + dev_size
    if required > len(train_dataset):
        raise ValueError(
            f"请求的 train_size + dev_size = {required}，"
            f"但官方 train split 只有 {len(train_dataset)} 条样本。"
        )

    shuffled_indices = list(range(len(train_dataset)))
    random.Random(seed).shuffle(shuffled_indices)

    train_indices = shuffled_indices[:train_size]
    dev_indices = shuffled_indices[train_size : train_size + dev_size]

    return train_dataset.select(train_indices), train_dataset.select(dev_indices)


def maybe_sample_test(
    test_dataset: "Dataset",
    *,
    test_size: int | None,
    seed: int,
) -> "Dataset":
    """可选地对子集 test 做抽样。"""

    if test_size is None or test_size <= 0 or test_size >= len(test_dataset):
        return test_dataset

    shuffled_indices = list(range(len(test_dataset)))
    random.Random(seed).shuffle(shuffled_indices)
    return test_dataset.select(shuffled_indices[:test_size])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(
    *,
    dataset_path: Path,
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    train_size: int,
    dev_size: int,
    requested_test_size: int | None,
    prompt_template: str,
) -> dict[str, Any]:
    """生成摘要信息。"""

    return {
        "dataset_name": "gsm8k",
        "dataset_config": "main",
        "source_dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_train_size": train_size,
        "requested_dev_size": dev_size,
        "requested_test_size": requested_test_size,
        "actual_sizes": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
        },
        "prompt_template": prompt_template,
        "fields": [
            "problem_id",
            "dataset_name",
            "dataset_config",
            "split",
            "source_index",
            "question",
            "solution",
            "answer",
            "final_answer",
            "final_answer_numeric",
            "answer_tag",
            "prompt",
        ],
    }


def run_self_check() -> int:

    example = {
        "question": "If John has 40 apples and gives away 8, how many are left?",
        "answer": "John starts with 40 apples and gives away 8, so 40 - 8 = 32.\n#### 32",
    }
    prepared = prepare_record(
        example,
        split="train",
        source_index=7,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    )
    payload = asdict(prepared)

    checks = {
        "problem_id": payload["problem_id"] == "gsm8k-main-train-00007",
        "final_answer": payload["final_answer"] == "32",
        "final_answer_numeric": payload["final_answer_numeric"] == "32",
        "answer_tag": payload["answer_tag"] == "<answer>32</answer>",
        "prompt_contains_question": "40 apples" in payload["prompt"],
    }

    print(json.dumps({"prepared_example": payload}, ensure_ascii=False))
    print(json.dumps({"checks": checks}, ensure_ascii=False))
    passed = all(checks.values())
    print(f"self_check_passed={passed}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""

    parser = argparse.ArgumentParser(description="Prepare GSM8K core experiment set")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/gsm8k/main"),
        help="GSM8K save_to_disk 目录，默认使用服务器上的推荐路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/gsm8k/core"),
        help="导出核心实验集的目录。",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1500,
        help="从官方 train split 抽取的训练样本数，默认 1500。",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=200,
        help="从官方 train split 中额外切出的开发集大小，默认 200。",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="可选的 test 抽样大小；<=0 表示保留完整官方 test。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )
    parser.add_argument(
        "--prompt-template-file",
        type=Path,
        help="自定义 prompt 模板文件；文件中需包含 `{question}` 占位符。",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="运行内置自检，不读取真实数据集。",
    )
    return parser


def main() -> int:
    """CLI 主入口。"""

    parser = build_parser()
    args = parser.parse_args()

    if args.self_check:
        return run_self_check()

    from datasets import DatasetDict, load_from_disk

    prompt_template = DEFAULT_PROMPT_TEMPLATE
    if args.prompt_template_file is not None:
        prompt_template = args.prompt_template_file.read_text(encoding="utf-8")
    if "{question}" not in prompt_template:
        raise ValueError("prompt 模板必须包含 `{question}` 占位符。")

    dataset = load_from_disk(str(args.dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError("dataset_path 必须指向包含 train/test split 的 DatasetDict。")
    if "train" not in dataset or "test" not in dataset:
        raise KeyError("GSM8K 数据集中必须包含 `train` 与 `test` split。")

    train_subset, dev_subset = sample_train_and_dev(
        dataset["train"],
        train_size=args.train_size,
        dev_size=args.dev_size,
        seed=args.seed,
    )
    test_subset = maybe_sample_test(
        dataset["test"],
        test_size=args.test_size if args.test_size > 0 else None,
        seed=args.seed,
    )

    train_rows = dataset_to_records(train_subset, split="train", prompt_template=prompt_template)
    dev_rows = dataset_to_records(dev_subset, split="dev", prompt_template=prompt_template)
    test_rows = dataset_to_records(test_subset, split="test", prompt_template=prompt_template)

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "dev.jsonl", dev_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    summary = build_summary(
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
        seed=args.seed,
        train_size=args.train_size,
        dev_size=args.dev_size,
        requested_test_size=args.test_size if args.test_size > 0 else None,
        prompt_template=prompt_template,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
