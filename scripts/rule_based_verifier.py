"""GSM8K 规则校验器。

这版先服务于阶段 1：
1. 从模型输出中抽取 `<answer>...</answer>` 的最终答案
2. 从 GSM8K 标注的 `#### final_answer` 中抽取标准答案
3. 对数值答案做归一化与严格比较

"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Iterable


ANSWER_TAG_PATTERN = re.compile(
    r"<answer>\s*(?P<answer>.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)

GSM8K_REFERENCE_PATTERN = re.compile(r"####\s*(?P<answer>.+?)\s*$", flags=re.MULTILINE)

NUMBER_PATTERN = re.compile(
    r"""
    (?P<number>
        [-+]?
        (?:
            (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
            |
            \.\d+
            |
            (?:\d{1,3}(?:,\d{3})+|\d+)\s*/\s*(?:\d{1,3}(?:,\d{3})+|\d+)
        )
        %?
    )
    """,
    flags=re.VERBOSE,
)


@dataclass(frozen=True)
class VerificationResult:
    """一次样本校验的结构化结果。"""

    is_correct: bool
    extracted_prediction: str | None
    extracted_reference: str | None
    comparison_mode: str
    reason: str


def extract_answer_tag(text: str) -> str | None:
    """优先抽取 `<answer>...</answer>` 中的内容。"""
    match = ANSWER_TAG_PATTERN.search(text)
    if not match:
        return None
    return match.group("answer").strip()


def extract_gsm8k_reference(text: str) -> str | None:
    """从 GSM8K 标注答案中抽取 `####` 后面的最终答案。"""
    match = GSM8K_REFERENCE_PATTERN.search(text)
    if not match:
        return None
    return match.group("answer").strip()


def extract_last_number(text: str) -> str | None:
    """抽取文本中的最后一个数值片段。

    这主要用于两种场景：
    1. `<answer>` 标签内部不够干净，例如 `<answer>72 apples</answer>`
    2. 非严格模式下，模型忘了输出 `<answer>` 标签
    """

    matches = list(NUMBER_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group("number").strip()


def normalize_whitespace(text: str) -> str:
    """统一空白字符，避免纯格式差异影响比较。"""
    return " ".join(text.strip().split())


def strip_outer_punctuation(text: str) -> str:
    """剥离答案外层常见噪声符号。"""
    stripped = text.strip()
    while stripped and stripped[-1] in ".。!！?？,，;；:":
        stripped = stripped[:-1].rstrip()
    while stripped and stripped[0] in "$￥£€:：":
        stripped = stripped[1:].lstrip()
    return stripped


def canonicalize_text_answer(text: str) -> str:
    """对非数值答案做轻量归一化。"""
    normalized = normalize_whitespace(text)
    normalized = strip_outer_punctuation(normalized)
    return normalized.casefold()


def normalize_numeric_string(text: str) -> str:
    """把数值字符串整理成更易解析的形式。"""
    normalized = normalize_whitespace(text)
    normalized = strip_outer_punctuation(normalized)
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("，", ",")
    normalized = normalized.replace("．", ".")
    normalized = normalized.replace("／", "/")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("￥", "")
    normalized = normalized.replace("£", "")
    normalized = normalized.replace("€", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace(" ", "")
    return normalized


def parse_numeric_answer(text: str) -> Decimal | None:
    """尝试把答案解析成可精确比较的数值。

    支持：
    - 整数 / 小数
    - 分数，如 `1/2`
    - 百分数，如 `12.5%`
    """

    normalized = normalize_numeric_string(text)
    if not normalized:
        return None

    is_percent = normalized.endswith("%")
    if is_percent:
        normalized = normalized[:-1]

    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", maxsplit=1)
            fraction = Fraction(numerator) / Fraction(denominator)
            value = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        else:
            value = Decimal(normalized)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None

    if is_percent:
        value /= Decimal("100")

    return value


def prepare_prediction_answer(prediction_text: str, strict: bool = True) -> tuple[str | None, str]:
    """抽取模型答案。

    返回：
    - 抽取到的答案字符串
    - 抽取模式说明
    """

    tagged_answer = extract_answer_tag(prediction_text)
    if tagged_answer is not None:
        return tagged_answer, "tag"

    if strict:
        return None, "missing_answer_tag"

    fallback_answer = extract_last_number(prediction_text)
    if fallback_answer is not None:
        return fallback_answer, "fallback_last_number"

    return None, "missing_prediction_answer"


def verify_gsm8k_prediction(
    prediction_text: str,
    reference_text: str,
    *,
    strict: bool = True,
) -> VerificationResult:
    """校验单条 GSM8K 预测结果是否正确。"""

    extracted_reference = extract_gsm8k_reference(reference_text)
    if extracted_reference is None:
        extracted_reference = normalize_whitespace(reference_text)

    extracted_prediction, prediction_mode = prepare_prediction_answer(
        prediction_text,
        strict=strict,
    )

    if extracted_prediction is None:
        return VerificationResult(
            is_correct=False,
            extracted_prediction=None,
            extracted_reference=extracted_reference,
            comparison_mode=prediction_mode,
            reason="未能从模型输出中抽取最终答案。",
        )

    prediction_number = parse_numeric_answer(extracted_prediction)
    reference_number = parse_numeric_answer(extracted_reference)

    # 标签内部常会混入单位或补充说明，例如 `72 apples`。
    # 只要能稳定抽到唯一的最终数值，就仍然按数值答案处理。
    if prediction_number is None:
        numeric_candidate = extract_last_number(extracted_prediction)
        if numeric_candidate is not None:
            prediction_number = parse_numeric_answer(numeric_candidate)

    if reference_number is None:
        numeric_candidate = extract_last_number(extracted_reference)
        if numeric_candidate is not None:
            reference_number = parse_numeric_answer(numeric_candidate)

    # GSM8K 默认优先走数值比较；两边都无法解析成数值时，再退回文本比较。
    if prediction_number is not None and reference_number is not None:
        is_correct = prediction_number == reference_number
        return VerificationResult(
            is_correct=is_correct,
            extracted_prediction=extracted_prediction,
            extracted_reference=extracted_reference,
            comparison_mode=f"{prediction_mode}_numeric",
            reason="数值比较完成。",
        )

    normalized_prediction = canonicalize_text_answer(extracted_prediction)
    normalized_reference = canonicalize_text_answer(extracted_reference)
    is_correct = normalized_prediction == normalized_reference
    return VerificationResult(
        is_correct=is_correct,
        extracted_prediction=extracted_prediction,
        extracted_reference=extracted_reference,
        comparison_mode=f"{prediction_mode}_text",
        reason="文本比较完成。",
    )


def iter_jsonl(path: Path) -> Iterable[dict]:
    """逐行读取 JSONL。"""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行解析失败: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 第 {line_number} 行不是对象。")
            yield record


def evaluate_jsonl(
    path: Path,
    *,
    prediction_field: str,
    reference_field: str,
    strict: bool,
) -> tuple[int, int]:
    """批量评测 JSONL 文件。"""

    total = 0
    correct = 0

    for record in iter_jsonl(path):
        total += 1
        prediction = str(record[prediction_field])
        reference = str(record[reference_field])
        result = verify_gsm8k_prediction(prediction, reference, strict=strict)
        correct += int(result.is_correct)

        print(
            json.dumps(
                {
                    "index": total - 1,
                    "is_correct": result.is_correct,
                    "prediction_answer": result.extracted_prediction,
                    "reference_answer": result.extracted_reference,
                    "comparison_mode": result.comparison_mode,
                    "reason": result.reason,
                },
                ensure_ascii=False,
            )
        )

    return correct, total


def run_self_check() -> int:
    """跑一组轻量自检样例，确认常见格式都能正确处理。"""

    examples = [
        {
            "prediction": "推导略。<answer>72</answer>",
            "reference": "We compute carefully. #### 72",
            "expected": True,
        },
        {
            "prediction": "推导略。<answer>1,234</answer>",
            "reference": "Reasoning #### 1234",
            "expected": True,
        },
        {
            "prediction": "推导略。<answer>72 apples</answer>",
            "reference": "Reasoning #### 72",
            "expected": True,
        },
        {
            "prediction": "推导略。<answer>1/2</answer>",
            "reference": "Reasoning #### 0.5",
            "expected": True,
        },
        {
            "prediction": "推导略。最终答案是 99",
            "reference": "Reasoning #### 99",
            "expected": False,
        },
        {
            "prediction": "推导略。<answer>98</answer>",
            "reference": "Reasoning #### 99",
            "expected": False,
        },
    ]

    all_ok = True
    for index, example in enumerate(examples, start=1):
        result = verify_gsm8k_prediction(
            example["prediction"],
            example["reference"],
            strict=True,
        )
        case_ok = result.is_correct == example["expected"]
        all_ok &= case_ok
        print(
            json.dumps(
                {
                    "case": index,
                    "expected": example["expected"],
                    "got": result.is_correct,
                    "comparison_mode": result.comparison_mode,
                    "prediction_answer": result.extracted_prediction,
                    "reference_answer": result.extracted_reference,
                    "reason": result.reason,
                },
                ensure_ascii=False,
            )
        )

    print(f"self_check_passed={all_ok}")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""

    parser = argparse.ArgumentParser(description="GSM8K Rule-based Verifier")
    parser.add_argument(
        "--prediction",
        help="单条模型输出文本。",
    )
    parser.add_argument(
        "--reference",
        help="单条参考答案文本；GSM8K 原始 answer 字段也可以直接传入。",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="批量评测的 JSONL 文件路径。",
    )
    parser.add_argument(
        "--prediction-field",
        default="prediction",
        help="JSONL 中模型输出字段名，默认是 prediction。",
    )
    parser.add_argument(
        "--reference-field",
        default="answer",
        help="JSONL 中参考答案字段名，默认是 answer。",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="允许在缺少 <answer> 标签时回退到“抽最后一个数”。",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="运行内置自检样例。",
    )
    return parser


def main() -> int:
    """CLI 主入口。"""

    parser = build_parser()
    args = parser.parse_args()
    strict = not args.allow_fallback

    if args.self_check:
        return run_self_check()

    if args.jsonl is not None:
        correct, total = evaluate_jsonl(
            args.jsonl,
            prediction_field=args.prediction_field,
            reference_field=args.reference_field,
            strict=strict,
        )
        accuracy = correct / total if total else 0.0
        print(
            json.dumps(
                {
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.prediction is None or args.reference is None:
        parser.error("单条校验需要同时提供 --prediction 和 --reference，或使用 --jsonl / --self-check。")

    result = verify_gsm8k_prediction(
        prediction_text=args.prediction,
        reference_text=args.reference,
        strict=strict,
    )
    print(
        json.dumps(
            {
                "is_correct": result.is_correct,
                "prediction_answer": result.extracted_prediction,
                "reference_answer": result.extracted_reference,
                "comparison_mode": result.comparison_mode,
                "reason": result.reason,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.is_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
