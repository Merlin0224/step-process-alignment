#!/usr/bin/env python
"""Convert Stage-2 MCTS trees into Step-DPO preference pairs.

This script reads the Stage-2 JSONL produced by `build_mcts_value_data.py`
and converts sibling node comparisons into pairwise preference samples.

Default behavior:
- use sibling nodes that share the same parent state
- compare them by `q_value`
- keep pairs whose score gap is at least `--min-score-gap`
- emit LLaMA-Factory friendly ShareGPT preference format

Supported output formats:
- `sharegpt`: conversations + chosen/rejected message objects
- `alpaca`: instruction/input + chosen/rejected strings
- `generic`: richer debugging payload for analysis
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_USER_PROMPT_SUFFIX = (
    "Continue from the current reasoning state and generate only the next best reasoning step. "
    "Do not restart the full solution. "
    "If this next step should finish the problem, put the final answer inside <answer>...</answer>."
)


@dataclass(frozen=True)
class PreferencePair:
    """A single step-level preference pair."""

    example_id: str
    problem_id: str
    parent_node_id: str
    parent_depth: int
    step_prompt: str
    parent_prefix_text: str
    chosen_node_id: str
    rejected_node_id: str
    chosen_text: str
    rejected_text: str
    chosen_score: float
    rejected_score: float
    score_gap: float
    score_field: str
    chosen_depth: int
    rejected_depth: int
    chosen_evaluation_mode: str | None
    rejected_evaluation_mode: str | None


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object.")
            rows.append(record)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text_block(text: str) -> str:
    """Normalize line endings and trim whitespace."""

    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def get_score(node: dict[str, Any], score_field: str) -> float | None:
    """Get the comparison score for a node."""

    value = node.get(score_field)
    if value is None:
        return None
    return float(value)


def build_step_prompt(
    record: dict[str, Any],
    parent_node: dict[str, Any],
    *,
    suffix: str,
) -> str:
    """Build the user-side prompt representing the current reasoning state."""

    base_prompt = normalize_text_block(record.get("prompt", ""))
    question = normalize_text_block(record.get("question", ""))
    parent_prefix = normalize_text_block(parent_node.get("prefix_text", ""))

    if not base_prompt:
        if not question:
            raise ValueError("Record must contain either `prompt` or `question`.")
        base_prompt = (
            "You are a careful math reasoning assistant.\n"
            "Solve the problem step by step.\n"
            "Put the final answer inside <answer>...</answer>.\n\n"
            f"Question:\n{question}"
        )

    if parent_prefix:
        state_text = parent_prefix
    else:
        state_text = "(empty)"

    return (
        f"{base_prompt}\n\n"
        f"Current reasoning state:\n{state_text}\n\n"
        f"{suffix.strip()}"
    ).strip()


def build_pair_record(
    pair: PreferencePair,
    *,
    output_format: str,
) -> dict[str, Any]:
    """Render one preference pair into the requested output format."""

    common_meta = {
        "id": pair.example_id,
        "problem_id": pair.problem_id,
        "parent_node_id": pair.parent_node_id,
        "parent_depth": pair.parent_depth,
        "parent_prefix_text": pair.parent_prefix_text,
        "chosen_node_id": pair.chosen_node_id,
        "rejected_node_id": pair.rejected_node_id,
        "chosen_score": pair.chosen_score,
        "rejected_score": pair.rejected_score,
        "score_gap": pair.score_gap,
        "score_field": pair.score_field,
        "chosen_depth": pair.chosen_depth,
        "rejected_depth": pair.rejected_depth,
        "chosen_evaluation_mode": pair.chosen_evaluation_mode,
        "rejected_evaluation_mode": pair.rejected_evaluation_mode,
    }

    if output_format == "sharegpt":
        return {
            "id": pair.example_id,
            "conversations": [
                {
                    "from": "human",
                    "value": pair.step_prompt,
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": pair.chosen_text,
            },
            "rejected": {
                "from": "gpt",
                "value": pair.rejected_text,
            },
            **common_meta,
        }

    if output_format == "alpaca":
        return {
            "instruction": pair.step_prompt,
            "input": "",
            "chosen": pair.chosen_text,
            "rejected": pair.rejected_text,
            **common_meta,
        }

    if output_format == "generic":
        return {
            "step_prompt": pair.step_prompt,
            "chosen": pair.chosen_text,
            "rejected": pair.rejected_text,
            **common_meta,
        }

    raise ValueError(f"Unsupported output format: {output_format}")


def make_dataset_info_snippet(output_path: Path, output_format: str) -> dict[str, Any] | None:
    """Build a `dataset_info.json` snippet for LLaMA-Factory."""

    dataset_name = output_path.stem
    if output_format == "sharegpt":
        return {
            dataset_name: {
                "file_name": output_path.name,
                "formatting": "sharegpt",
                "ranking": True,
                "columns": {
                    "messages": "conversations",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            }
        }

    if output_format == "alpaca":
        return {
            dataset_name: {
                "file_name": output_path.name,
                "ranking": True,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            }
        }

    return None


def convert_record_to_pairs(
    record: dict[str, Any],
    *,
    score_field: str,
    min_score_gap: float,
    prompt_suffix: str,
) -> tuple[list[PreferencePair], dict[str, int]]:
    """Convert one MCTS tree record into sibling preference pairs."""

    nodes = record.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return [], {"parent_groups": 0, "paired_groups": 0, "pairs": 0, "skipped_missing_score": 0, "skipped_same_text": 0, "skipped_small_gap": 0, "skipped_tie": 0}

    node_by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[dict[str, Any]]] = {}

    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        node_by_id[node_id] = node
        parent_id = node.get("parent_id")
        if isinstance(parent_id, str):
            children_by_parent.setdefault(parent_id, []).append(node)

    pairs: list[PreferencePair] = []
    stats = {
        "parent_groups": 0,
        "paired_groups": 0,
        "pairs": 0,
        "skipped_missing_score": 0,
        "skipped_same_text": 0,
        "skipped_small_gap": 0,
        "skipped_tie": 0,
    }

    for parent_id, children in children_by_parent.items():
        if len(children) < 2:
            continue

        parent_node = node_by_id.get(parent_id)
        if parent_node is None:
            continue

        stats["parent_groups"] += 1
        group_pair_count = 0
        step_prompt = build_step_prompt(record, parent_node, suffix=prompt_suffix)
        parent_prefix_text = normalize_text_block(parent_node.get("prefix_text", ""))

        for left, right in combinations(children, 2):
            left_score = get_score(left, score_field)
            right_score = get_score(right, score_field)
            if left_score is None or right_score is None:
                stats["skipped_missing_score"] += 1
                continue

            left_text = normalize_text_block(left.get("step_text", ""))
            right_text = normalize_text_block(right.get("step_text", ""))
            if not left_text or not right_text:
                stats["skipped_missing_score"] += 1
                continue
            if left_text == right_text:
                stats["skipped_same_text"] += 1
                continue

            score_gap = abs(left_score - right_score)
            if score_gap < min_score_gap:
                if score_gap == 0:
                    stats["skipped_tie"] += 1
                else:
                    stats["skipped_small_gap"] += 1
                continue

            if left_score > right_score:
                chosen_node = left
                rejected_node = right
                chosen_text = left_text
                rejected_text = right_text
                chosen_score = left_score
                rejected_score = right_score
            elif right_score > left_score:
                chosen_node = right
                rejected_node = left
                chosen_text = right_text
                rejected_text = left_text
                chosen_score = right_score
                rejected_score = left_score
            else:
                stats["skipped_tie"] += 1
                continue

            pair = PreferencePair(
                example_id=(
                    f"{record.get('problem_id', 'unknown')}"
                    f"__{parent_id}__{chosen_node.get('node_id')}__{rejected_node.get('node_id')}"
                ),
                problem_id=str(record.get("problem_id", "unknown")),
                parent_node_id=parent_id,
                parent_depth=int(parent_node.get("depth", 0)),
                step_prompt=step_prompt,
                parent_prefix_text=parent_prefix_text,
                chosen_node_id=str(chosen_node.get("node_id")),
                rejected_node_id=str(rejected_node.get("node_id")),
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                score_gap=chosen_score - rejected_score,
                score_field=score_field,
                chosen_depth=int(chosen_node.get("depth", 0)),
                rejected_depth=int(rejected_node.get("depth", 0)),
                chosen_evaluation_mode=chosen_node.get("evaluation_mode"),
                rejected_evaluation_mode=rejected_node.get("evaluation_mode"),
            )
            pairs.append(pair)
            group_pair_count += 1

        if group_pair_count > 0:
            stats["paired_groups"] += 1
            stats["pairs"] += group_pair_count

    return pairs, stats


def convert_records(
    records: list[dict[str, Any]],
    *,
    output_format: str,
    score_field: str,
    min_score_gap: float,
    prompt_suffix: str,
    max_pairs_per_problem: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert all tree records and summarize the conversion."""

    all_pairs: list[PreferencePair] = []
    total_parent_groups = 0
    total_paired_groups = 0
    skipped_missing_score = 0
    skipped_same_text = 0
    skipped_small_gap = 0
    skipped_tie = 0
    problems_with_pairs = 0
    pair_counts_per_problem: list[int] = []

    for record in records:
        record_pairs, record_stats = convert_record_to_pairs(
            record,
            score_field=score_field,
            min_score_gap=min_score_gap,
            prompt_suffix=prompt_suffix,
        )
        if max_pairs_per_problem is not None and max_pairs_per_problem > 0:
            record_pairs = record_pairs[:max_pairs_per_problem]

        pair_count = len(record_pairs)
        pair_counts_per_problem.append(pair_count)
        if pair_count > 0:
            problems_with_pairs += 1
            all_pairs.extend(record_pairs)

        total_parent_groups += record_stats["parent_groups"]
        total_paired_groups += record_stats["paired_groups"]
        skipped_missing_score += record_stats["skipped_missing_score"]
        skipped_same_text += record_stats["skipped_same_text"]
        skipped_small_gap += record_stats["skipped_small_gap"]
        skipped_tie += record_stats["skipped_tie"]

    rendered_rows = [build_pair_record(pair, output_format=output_format) for pair in all_pairs]

    chosen_scores = [pair.chosen_score for pair in all_pairs]
    rejected_scores = [pair.rejected_score for pair in all_pairs]
    score_gaps = [pair.score_gap for pair in all_pairs]
    parent_depths = [pair.parent_depth for pair in all_pairs]

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_problem_count": len(records),
        "output_pair_count": len(rendered_rows),
        "problems_with_pairs": problems_with_pairs,
        "problem_pair_ratio": (problems_with_pairs / len(records)) if records else 0.0,
        "score_field": score_field,
        "min_score_gap": min_score_gap,
        "output_format": output_format,
        "max_pairs_per_problem": max_pairs_per_problem,
        "parent_groups": total_parent_groups,
        "paired_parent_groups": total_paired_groups,
        "skipped_missing_score": skipped_missing_score,
        "skipped_same_text": skipped_same_text,
        "skipped_small_gap": skipped_small_gap,
        "skipped_tie": skipped_tie,
        "avg_pairs_per_problem": mean(pair_counts_per_problem) if pair_counts_per_problem else 0.0,
        "avg_chosen_score": mean(chosen_scores) if chosen_scores else 0.0,
        "avg_rejected_score": mean(rejected_scores) if rejected_scores else 0.0,
        "avg_score_gap": mean(score_gaps) if score_gaps else 0.0,
        "avg_parent_depth": mean(parent_depths) if parent_depths else 0.0,
        "example_fields": list(rendered_rows[0].keys()) if rendered_rows else [],
    }

    return rendered_rows, summary


def run_self_check() -> int:
    """Lightweight local sanity check."""

    toy_records = [
        {
            "problem_id": "gsm8k-main-train-00000",
            "question": "What is 2 + 2?",
            "prompt": (
                "You are a careful math reasoning assistant.\n"
                "Solve the problem step by step.\n"
                "Put the final answer inside <answer>...</answer>.\n\n"
                "Question:\nWhat is 2 + 2?"
            ),
            "nodes": [
                {
                    "node_id": "root",
                    "parent_id": None,
                    "depth": 0,
                    "prefix_text": "",
                    "step_text": "",
                    "q_value": None,
                    "mean_value": 0.0,
                },
                {
                    "node_id": "node-00001",
                    "parent_id": "root",
                    "depth": 1,
                    "prefix_text": "Step 1: Add the two numbers.",
                    "step_text": "Step 1: Add the two numbers.",
                    "q_value": 1.0,
                    "mean_value": 1.0,
                    "evaluation_mode": "rollout",
                },
                {
                    "node_id": "node-00002",
                    "parent_id": "root",
                    "depth": 1,
                    "prefix_text": "Step 1: Guess 5 immediately.",
                    "step_text": "Step 1: Guess 5 immediately.",
                    "q_value": 0.0,
                    "mean_value": 0.0,
                    "evaluation_mode": "rollout",
                },
                {
                    "node_id": "node-00003",
                    "parent_id": "node-00001",
                    "depth": 2,
                    "prefix_text": "Step 1: Add the two numbers.\nStep 2: 2 + 2 = 4.\n<answer>4</answer>",
                    "step_text": "Step 2: 2 + 2 = 4.\n<answer>4</answer>",
                    "q_value": 1.0,
                    "mean_value": 1.0,
                    "evaluation_mode": "terminal",
                },
                {
                    "node_id": "node-00004",
                    "parent_id": "node-00001",
                    "depth": 2,
                    "prefix_text": "Step 1: Add the two numbers.\nStep 2: 2 + 2 = 3.\n<answer>3</answer>",
                    "step_text": "Step 2: 2 + 2 = 3.\n<answer>3</answer>",
                    "q_value": 0.0,
                    "mean_value": 0.0,
                    "evaluation_mode": "terminal",
                },
            ],
        }
    ]

    rows, summary = convert_records(
        toy_records,
        output_format="sharegpt",
        score_field="q_value",
        min_score_gap=0.2,
        prompt_suffix=DEFAULT_USER_PROMPT_SUFFIX,
        max_pairs_per_problem=None,
    )

    checks = {
        "pair_count": summary["output_pair_count"] == 2,
        "problem_with_pairs": summary["problems_with_pairs"] == 1,
        "sharegpt_messages": rows[0]["conversations"][0]["from"] == "human",
        "chosen_is_better": rows[0]["chosen_score"] > rows[0]["rejected_score"],
        "contains_current_state": "Current reasoning state:" in rows[0]["conversations"][0]["value"],
    }

    print(json.dumps({"checks": checks, "summary": summary}, ensure_ascii=False))
    passed = all(checks.values())
    print(f"self_check_passed={passed}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Prepare Step-DPO preference pairs from Stage-2 MCTS trees")
    parser.add_argument("--input-jsonl", type=Path, help="Stage-2 MCTS JSONL produced by build_mcts_value_data.py")
    parser.add_argument("--output-jsonl", type=Path, help="Output JSONL for DPO preference training")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <output-jsonl>.summary.json",
    )
    parser.add_argument(
        "--output-format",
        choices=("sharegpt", "alpaca", "generic"),
        default="sharegpt",
        help="Which dataset schema to emit",
    )
    parser.add_argument(
        "--score-field",
        choices=("q_value", "mean_value"),
        default="q_value",
        help="Which node score to compare",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.2,
        help="Minimum absolute score gap required to keep a pair",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optionally limit the number of input problems",
    )
    parser.add_argument(
        "--max-pairs-per-problem",
        type=int,
        default=None,
        help="Optionally cap emitted pairs per problem after sorting order is preserved",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default=DEFAULT_USER_PROMPT_SUFFIX,
        help="Instruction suffix appended after the current reasoning state",
    )
    parser.add_argument("--self-check", action="store_true", help="Run a lightweight self-check")
    return parser


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    if args.self_check:
        return run_self_check()

    if args.input_jsonl is None or args.output_jsonl is None:
        parser.error("--input-jsonl and --output-jsonl are required unless --self-check is used.")

    records = iter_jsonl(args.input_jsonl)
    if args.max_problems is not None:
        if args.max_problems <= 0:
            raise ValueError("--max-problems must be positive when provided.")
        records = records[: args.max_problems]

    rows, summary = convert_records(
        records,
        output_format=args.output_format,
        score_field=args.score_field,
        min_score_gap=args.min_score_gap,
        prompt_suffix=args.prompt_suffix,
        max_pairs_per_problem=args.max_pairs_per_problem,
    )

    summary_json = args.summary_json
    if summary_json is None:
        summary_json = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")

    summary["input_jsonl"] = str(args.input_jsonl)
    summary["output_jsonl"] = str(args.output_jsonl)
    summary["summary_json"] = str(summary_json)
    summary["dataset_info_snippet"] = make_dataset_info_snippet(args.output_jsonl, args.output_format)

    write_jsonl(args.output_jsonl, rows)
    write_json(summary_json, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
