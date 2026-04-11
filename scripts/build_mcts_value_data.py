#!/usr/bin/env python
"""阶段 2：MCTS 数据生成与步级价值估计骨架。

第一版目标不是一次性做完“最强搜索器”，而是先把下面几件事稳定打通：
1. 从核心实验集 JSONL 读取问题
2. 采样初始推理路径，并按“步”切分中间状态
3. 对每个中间状态做 Monte Carlo rollout，估计步级价值 Q_t
4. 持久化树状数据，供后续 Step-DPO / Step-GRPO 复用

这版刻意采用“前缀回放 + rollout 估值”的保守实现：
- 工程上更稳
- 更容易排查显存与生成问题
- 之后可以自然扩展到更完整的 UCT / PUCT 选择策略
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    from rule_based_verifier import extract_answer_tag, verify_gsm8k_prediction
except ModuleNotFoundError:  # pragma: no cover - 兼容以 scripts.build_mcts_value_data 方式导入
    from scripts.rule_based_verifier import extract_answer_tag, verify_gsm8k_prediction


@dataclass(frozen=True)
class SamplingConfig:
    """生成采样配置。"""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateBoundaryConfig:
    """状态切分与截断配置。"""

    line_break_delimiter: str = "\n"
    regex_pattern: str = r"(?=Step\s+\d+\s*:)"
    max_step_chars: int = 512
    max_step_tokens: int | None = None
    tokenizer_name_or_path: str | None = None
    allow_regex_split: bool = True


@dataclass(frozen=True)
class GenerationResponse:
    """一次生成请求的最小返回结构。"""

    text: str
    finish_reason: str | None = None


@dataclass
class RolloutRecord:
    """某个前缀的一次 rollout 结果。"""

    rollout_index: int
    continuation_text: str
    full_prediction: str
    is_correct: bool
    comparison_mode: str
    extracted_prediction: str | None


@dataclass
class ValueEstimate:
    """某个前缀的步级价值估计。"""

    q_value: float
    success_count: int
    rollout_count: int
    evaluation_mode: str
    terminal_correct: bool | None
    rollout_records: list[RolloutRecord] = field(default_factory=list)


@dataclass
class SearchSimulationEvent:
    """记录一次搜索迭代的关键动作。"""

    simulation_index: int
    selected_path: list[str]
    expanded_node_id: str | None
    expanded_child_ids: list[str]
    backed_up_value: float
    event_type: str


@dataclass
class SearchTreeNode:
    """树上的一个中间状态节点。"""

    node_id: str
    parent_id: str | None
    depth: int
    step_text: str
    prefix_text: str
    step_path: tuple[str, ...] = field(repr=False)
    q_value: float | None
    success_count: int
    rollout_count: int
    evaluation_mode: str
    terminal_correct: bool | None
    path_hits: int = 1
    visit_count: int = 0
    total_value: float = 0.0
    children_ids: list[str] = field(default_factory=list)
    is_expanded: bool = False
    is_terminal: bool = False
    rollout_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if self.visit_count <= 0:
            return self.q_value if self.q_value is not None else 0.0
        return self.total_value / self.visit_count


class GenerationBackend(Protocol):
    """可插拔的生成后端协议。"""

    async def generate_batch(
        self,
        prompts: list[str],
        sampling_config: SamplingConfig,
    ) -> list[GenerationResponse]:
        """对一组 prompt 做生成。"""

    async def aclose(self) -> None:
        """释放资源。"""


def normalize_text_block(text: str) -> str:
    """清理首尾空白，但尽量不破坏行结构。"""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


class TokenizerHelper:
    """按需加载 tokenizer，并提供轻量 token 截断能力。"""

    def __init__(self, tokenizer_name_or_path: str | None) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self._tokenizer: Any | None = None

    def _ensure_tokenizer(self) -> Any:
        if self.tokenizer_name_or_path is None:
            return None
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def truncate_text(self, text: str, max_tokens: int | None) -> str:
        if max_tokens is None or max_tokens <= 0:
            return text
        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            return text
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).strip()


class ReasoningStateSegmenter:
    """统一管理步级状态切分与单步截断。"""

    def __init__(self, config: StateBoundaryConfig) -> None:
        self.config = config
        self.tokenizer_helper = TokenizerHelper(config.tokenizer_name_or_path)

    def _split_by_lines(self, text: str) -> list[str]:
        delimiter = self.config.line_break_delimiter
        return [line.strip() for line in text.split(delimiter) if line.strip()]

    def _split_by_regex(self, text: str) -> list[str]:
        if not self.config.allow_regex_split or not self.config.regex_pattern:
            return []
        matches = list(re.finditer(self.config.regex_pattern, text, flags=re.IGNORECASE))
        if len(matches) <= 1:
            return []
        indices = [match.start() for match in matches] + [len(text)]
        pieces: list[str] = []
        for start, end in zip(indices[:-1], indices[1:]):
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
        return pieces

    def _truncate_single_step(self, step_text: str) -> str:
        clipped = step_text.strip()
        if self.config.max_step_chars > 0:
            clipped = clipped[: self.config.max_step_chars].strip()
        clipped = self.tokenizer_helper.truncate_text(clipped, self.config.max_step_tokens)
        return clipped.strip()

    def split_steps(self, text: str) -> list[str]:
        normalized = normalize_text_block(text)
        if not normalized:
            return []

        regex_steps = self._split_by_regex(normalized)
        if regex_steps:
            return [step for step in (self._truncate_single_step(piece) for piece in regex_steps) if step]

        raw_lines = self._split_by_lines(normalized)
        if len(raw_lines) > 1:
            return [step for step in (self._truncate_single_step(piece) for piece in raw_lines) if step]

        single_step = self._truncate_single_step(normalized)
        return [single_step] if single_step else []

    def extract_first_step(self, generated_text: str) -> str | None:
        steps = self.split_steps(generated_text)
        if not steps:
            return None
        return steps[0]


def join_steps(steps: list[str]) -> str:
    """把多步推理还原成前缀文本。"""
    return "\n".join(step.strip() for step in steps if step.strip()).strip()


def build_continuation_prompt(base_prompt: str, prefix_text: str) -> str:
    """把问题 prompt 与当前前缀拼成下一次生成的输入。"""
    prompt = base_prompt.rstrip()
    if not prefix_text.strip():
        return prompt + "\n"
    return prompt + "\n" + prefix_text.rstrip() + "\n"


def merge_prefix_and_continuation(prefix_text: str, continuation_text: str) -> str:
    """把前缀与 rollout 后缀合并成完整预测。"""
    prefix = prefix_text.rstrip()
    continuation = continuation_text.strip()
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    return prefix + "\n" + continuation


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 到内存。

    阶段 2 初期样本规模不大，这里优先保证代码清晰。
    """

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 第 {line_number} 行不是对象。")
            rows.append(record)
    return rows


class MockGenerationBackend:
    """本地自检用的可脚本化后端。"""

    def __init__(self, scripted_outputs: dict[str, list[str]]) -> None:
        self.scripted_outputs = dict(scripted_outputs)
        self.call_counters: dict[str, int] = {key: 0 for key in scripted_outputs}

    def _match_key(self, prompt: str) -> str:
        candidates = [key for key in self.scripted_outputs if key in prompt]
        if not candidates:
            return "__default__"
        return max(candidates, key=len)

    async def generate_batch(
        self,
        prompts: list[str],
        sampling_config: SamplingConfig,
    ) -> list[GenerationResponse]:
        del sampling_config
        responses: list[GenerationResponse] = []
        for prompt in prompts:
            key = self._match_key(prompt)
            outputs = self.scripted_outputs.get(key, self.scripted_outputs.get("__default__", ["<answer>0</answer>"]))
            index = self.call_counters.setdefault(key, 0)
            text = outputs[index % len(outputs)]
            self.call_counters[key] = index + 1
            responses.append(GenerationResponse(text=text, finish_reason="mock"))
        await asyncio.sleep(0)
        return responses

    async def aclose(self) -> None:
        return None


class VLLMAsyncGenerationBackend:
    """vLLM AsyncLLMEngine 后端。

    这里按 vLLM 0.8.3 官方文档使用：
    - `AsyncLLMEngine.from_engine_args(...)`
    - `engine.generate(...)` 返回异步迭代器
    - 清理时调用 `shutdown_background_loop()`
    """

    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.enable_prefix_caching = enable_prefix_caching
        self._engine: Any | None = None

    async def _ensure_engine(self) -> None:
        if self._engine is not None:
            return

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            enable_prefix_caching=self.enable_prefix_caching,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=True)

    async def _generate_one(self, prompt: str, sampling_config: SamplingConfig) -> GenerationResponse:
        from vllm import SamplingParams

        await self._ensure_engine()
        assert self._engine is not None

        sampling_params = SamplingParams(
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            max_tokens=sampling_config.max_tokens,
            stop=list(sampling_config.stop) if sampling_config.stop else None,
        )

        request_id = str(uuid.uuid4())
        final_output = None
        async for request_output in self._engine.generate(prompt, sampling_params, request_id):
            final_output = request_output

        if final_output is None or not final_output.outputs:
            return GenerationResponse(text="", finish_reason="empty")

        output = final_output.outputs[0]
        return GenerationResponse(
            text=output.text,
            finish_reason=getattr(output, "finish_reason", None),
        )

    async def generate_batch(
        self,
        prompts: list[str],
        sampling_config: SamplingConfig,
    ) -> list[GenerationResponse]:
        tasks = [self._generate_one(prompt, sampling_config) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def aclose(self) -> None:
        if self._engine is None:
            return
        shutdown = getattr(self._engine, "shutdown_background_loop", None)
        if callable(shutdown):
            shutdown()
        self._engine = None


class StepValueMCTSEngine:
    """步级价值数据生成引擎。"""

    def __init__(
        self,
        *,
        backend: GenerationBackend,
        root_expansion_branches: int,
        expansion_branches: int,
        rollout_samples: int,
        max_step_depth: int,
        step_sampling_config: SamplingConfig,
        rollout_sampling_config: SamplingConfig,
        state_segmenter: ReasoningStateSegmenter,
        num_simulations: int,
        store_rollouts: bool,
        ucb_c: float,
    ) -> None:
        self.backend = backend
        self.root_expansion_branches = root_expansion_branches
        self.expansion_branches = expansion_branches
        self.rollout_samples = rollout_samples
        self.max_step_depth = max_step_depth
        self.step_sampling_config = step_sampling_config
        self.rollout_sampling_config = rollout_sampling_config
        self.state_segmenter = state_segmenter
        self.num_simulations = num_simulations
        self.store_rollouts = store_rollouts
        self.ucb_c = ucb_c

    async def estimate_prefix_value(self, record: dict[str, Any], prefix_text: str) -> ValueEstimate:
        """估计某个中间状态的 Q_t。"""

        if extract_answer_tag(prefix_text) is not None:
            terminal_result = verify_gsm8k_prediction(prefix_text, record["answer"], strict=True)
            return ValueEstimate(
                q_value=float(terminal_result.is_correct),
                success_count=int(terminal_result.is_correct),
                rollout_count=1,
                evaluation_mode="terminal",
                terminal_correct=terminal_result.is_correct,
                rollout_records=[],
            )

        prompt = build_continuation_prompt(record["prompt"], prefix_text)
        prompts = [prompt for _ in range(self.rollout_samples)]
        generations = await self.backend.generate_batch(prompts, self.rollout_sampling_config)

        rollout_records: list[RolloutRecord] = []
        success_count = 0
        for rollout_index, generation in enumerate(generations):
            full_prediction = merge_prefix_and_continuation(prefix_text, generation.text)
            result = verify_gsm8k_prediction(full_prediction, record["answer"], strict=True)
            success_count += int(result.is_correct)
            rollout_records.append(
                RolloutRecord(
                    rollout_index=rollout_index,
                    continuation_text=generation.text,
                    full_prediction=full_prediction,
                    is_correct=result.is_correct,
                    comparison_mode=result.comparison_mode,
                    extracted_prediction=result.extracted_prediction,
                )
            )

        q_value = success_count / max(len(rollout_records), 1)
        return ValueEstimate(
            q_value=q_value,
            success_count=success_count,
            rollout_count=len(rollout_records),
            evaluation_mode="rollout",
            terminal_correct=None,
            rollout_records=rollout_records if self.store_rollouts else [],
        )

    def _make_root_node(self) -> SearchTreeNode:
        return SearchTreeNode(
            node_id="root",
            parent_id=None,
            depth=0,
            step_text="",
            prefix_text="",
            step_path=(),
            q_value=None,
            success_count=0,
            rollout_count=0,
            evaluation_mode="root",
            terminal_correct=None,
            path_hits=1,
            visit_count=0,
            total_value=0.0,
            children_ids=[],
            is_expanded=False,
            is_terminal=False,
            rollout_records=[],
        )

    def _node_to_payload(self, node: SearchTreeNode) -> dict[str, Any]:
        payload = asdict(node)
        payload["mean_value"] = node.mean_value
        payload["step_path"] = list(node.step_path)
        return payload

    def _ucb_score(self, parent: SearchTreeNode, child: SearchTreeNode) -> float:
        prior = child.q_value if child.q_value is not None else 0.0
        if child.visit_count <= 0:
            return prior + self.ucb_c * math.sqrt(math.log(parent.visit_count + 2))
        exploitation = child.mean_value
        exploration = self.ucb_c * math.sqrt(math.log(parent.visit_count + 2) / child.visit_count)
        return exploitation + exploration + 0.05 * prior

    def _select_path(
        self,
        nodes: list[SearchTreeNode],
        node_id_to_index: dict[str, int],
    ) -> list[int]:
        path_indices = [0]
        current_index = 0
        while True:
            current_node = nodes[current_index]
            if current_node.is_terminal or current_node.depth >= self.max_step_depth:
                return path_indices
            if not current_node.is_expanded or not current_node.children_ids:
                return path_indices
            child_indices = [node_id_to_index[child_id] for child_id in current_node.children_ids]
            next_index = max(
                child_indices,
                key=lambda child_index: self._ucb_score(current_node, nodes[child_index]),
            )
            path_indices.append(next_index)
            current_index = next_index

    def _backup(
        self,
        nodes: list[SearchTreeNode],
        path_indices: list[int],
        value: float,
    ) -> None:
        for node_index in path_indices:
            nodes[node_index].visit_count += 1
            nodes[node_index].total_value += value

    async def _expand_node(
        self,
        *,
        record: dict[str, Any],
        node_index: int,
        nodes: list[SearchTreeNode],
        prefix_to_node_index: dict[tuple[str, ...], int],
        node_id_to_index: dict[str, int],
        branch_count: int,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        node = nodes[node_index]
        if node.is_terminal or node.depth >= self.max_step_depth:
            node.is_expanded = True
            return [], []

        prompt = build_continuation_prompt(record["prompt"], node.prefix_text)
        generations = await self.backend.generate_batch(
            [prompt for _ in range(branch_count)],
            self.step_sampling_config,
        )

        child_indices: list[int] = []
        expansion_records: list[dict[str, Any]] = []
        for branch_index, generation in enumerate(generations):
            normalized_completion = normalize_text_block(generation.text)
            step_text = self.state_segmenter.extract_first_step(normalized_completion)
            if not step_text:
                expansion_records.append(
                    {
                        "branch_index": branch_index,
                        "raw_completion": generation.text,
                        "first_step_text": None,
                        "child_node_id": None,
                        "status": "empty_step",
                    }
                )
                continue

            child_step_path = node.step_path + (step_text,)
            if child_step_path in prefix_to_node_index:
                child_index = prefix_to_node_index[child_step_path]
                child_node = nodes[child_index]
                child_node.path_hits += 1
                if child_node.node_id not in node.children_ids:
                    node.children_ids.append(child_node.node_id)
                child_indices.append(child_index)
                expansion_records.append(
                    {
                        "branch_index": branch_index,
                        "raw_completion": generation.text,
                        "first_step_text": step_text,
                        "child_node_id": child_node.node_id,
                        "status": "reused",
                        "q_value": child_node.q_value,
                    }
                )
                continue

            child_prefix_text = join_steps(list(child_step_path))
            estimate = await self.estimate_prefix_value(record, child_prefix_text)
            child_node = SearchTreeNode(
                node_id=f"node-{len(nodes):05d}",
                parent_id=node.node_id,
                depth=node.depth + 1,
                step_text=step_text,
                prefix_text=child_prefix_text,
                step_path=child_step_path,
                q_value=estimate.q_value,
                success_count=estimate.success_count,
                rollout_count=estimate.rollout_count,
                evaluation_mode=estimate.evaluation_mode,
                terminal_correct=estimate.terminal_correct,
                path_hits=1,
                visit_count=0,
                total_value=0.0,
                children_ids=[],
                is_expanded=False,
                is_terminal=(estimate.terminal_correct is not None) or (node.depth + 1 >= self.max_step_depth),
                rollout_records=[asdict(item) for item in estimate.rollout_records],
            )
            nodes.append(child_node)
            child_index = len(nodes) - 1
            prefix_to_node_index[child_step_path] = child_index
            node_id_to_index[child_node.node_id] = child_index
            node.children_ids.append(child_node.node_id)
            child_indices.append(child_index)
            expansion_records.append(
                {
                    "branch_index": branch_index,
                    "raw_completion": generation.text,
                    "first_step_text": step_text,
                    "child_node_id": child_node.node_id,
                    "status": "created",
                    "q_value": child_node.q_value,
                    "evaluation_mode": child_node.evaluation_mode,
                }
            )

        node.is_expanded = True
        return child_indices, expansion_records

    async def build_problem_tree(self, record: dict[str, Any]) -> dict[str, Any]:
        """围绕一条问题构建树状前缀数据。"""

        nodes: list[SearchTreeNode] = [self._make_root_node()]
        prefix_to_node_index: dict[tuple[str, ...], int] = {(): 0}
        node_id_to_index: dict[str, int] = {"root": 0}

        initial_child_indices, initial_paths = await self._expand_node(
            record=record,
            node_index=0,
            nodes=nodes,
            prefix_to_node_index=prefix_to_node_index,
            node_id_to_index=node_id_to_index,
            branch_count=self.root_expansion_branches,
        )
        for child_index in sorted(set(initial_child_indices)):
            child_value = nodes[child_index].q_value if nodes[child_index].q_value is not None else 0.0
            self._backup(nodes, [0, child_index], child_value)

        simulation_events: list[SearchSimulationEvent] = []
        for simulation_index in range(self.num_simulations):
            selected_path = self._select_path(nodes, node_id_to_index)
            leaf_index = selected_path[-1]
            leaf_node = nodes[leaf_index]
            selected_node_ids = [nodes[path_index].node_id for path_index in selected_path]

            if leaf_node.is_terminal or leaf_node.depth >= self.max_step_depth:
                backed_up_value = leaf_node.q_value if leaf_node.q_value is not None else leaf_node.mean_value
                self._backup(nodes, selected_path, backed_up_value)
                simulation_events.append(
                    SearchSimulationEvent(
                        simulation_index=simulation_index,
                        selected_path=selected_node_ids,
                        expanded_node_id=None,
                        expanded_child_ids=[],
                        backed_up_value=backed_up_value,
                        event_type="terminal_backup",
                    )
                )
                continue

            expanded_child_indices, _ = await self._expand_node(
                record=record,
                node_index=leaf_index,
                nodes=nodes,
                prefix_to_node_index=prefix_to_node_index,
                node_id_to_index=node_id_to_index,
                branch_count=self.expansion_branches,
            )
            if not expanded_child_indices:
                leaf_node.is_terminal = True
                backed_up_value = leaf_node.q_value if leaf_node.q_value is not None else 0.0
                self._backup(nodes, selected_path, backed_up_value)
                simulation_events.append(
                    SearchSimulationEvent(
                        simulation_index=simulation_index,
                        selected_path=selected_node_ids,
                        expanded_node_id=leaf_node.node_id,
                        expanded_child_ids=[],
                        backed_up_value=backed_up_value,
                        event_type="dead_end_backup",
                    )
                )
                continue

            best_child_index = max(
                expanded_child_indices,
                key=lambda child_index: (
                    nodes[child_index].q_value if nodes[child_index].q_value is not None else nodes[child_index].mean_value
                ),
            )
            best_child = nodes[best_child_index]
            backed_up_value = best_child.q_value if best_child.q_value is not None else best_child.mean_value
            self._backup(nodes, selected_path + [best_child_index], backed_up_value)
            simulation_events.append(
                SearchSimulationEvent(
                    simulation_index=simulation_index,
                    selected_path=selected_node_ids,
                    expanded_node_id=leaf_node.node_id,
                    expanded_child_ids=[nodes[child_index].node_id for child_index in expanded_child_indices],
                    backed_up_value=backed_up_value,
                    event_type="expand_backup",
                )
            )

        node_payloads = [self._node_to_payload(node) for node in nodes]
        max_q = max(
            (node["q_value"] for node in node_payloads if node["q_value"] is not None),
            default=None,
        )
        min_q = min(
            (node["q_value"] for node in node_payloads if node["q_value"] is not None),
            default=None,
        )

        return {
            "problem_id": record.get("problem_id"),
            "question": record.get("question"),
            "prompt": record.get("prompt"),
            "reference_answer": record.get("final_answer"),
            "reference_answer_numeric": record.get("final_answer_numeric"),
            "nodes": node_payloads,
            "initial_paths": initial_paths,
            "simulation_events": [asdict(item) for item in simulation_events],
            "statistics": {
                "node_count": len(node_payloads),
                "initial_branch_count": len(initial_paths),
                "simulation_count": len(simulation_events),
                "max_depth_reached": max((node["depth"] for node in node_payloads), default=0),
                "max_q_value": max_q,
                "min_q_value": min_q,
            },
        }


def make_backend(args: argparse.Namespace) -> GenerationBackend:
    """根据 CLI 选项构建后端。"""

    if args.backend == "mock":
        scripted_outputs = {
            "__default__": ["Reason briefly.\n<answer>0</answer>"],
        }
        return MockGenerationBackend(scripted_outputs)

    if args.backend == "vllm":
        return VLLMAsyncGenerationBackend(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=not args.no_trust_remote_code,
            enable_prefix_caching=not args.no_prefix_caching,
        )

    raise ValueError(f"不支持的 backend: {args.backend}")


def make_state_segmenter(args: argparse.Namespace) -> ReasoningStateSegmenter:
    """根据 CLI 参数构建状态切分器。"""

    tokenizer_name_or_path = args.state_tokenizer
    if tokenizer_name_or_path is None and args.state_max_step_tokens:
        tokenizer_name_or_path = args.model

    return ReasoningStateSegmenter(
        StateBoundaryConfig(
            regex_pattern=args.state_regex_pattern,
            max_step_chars=args.state_max_step_chars,
            max_step_tokens=args.state_max_step_tokens,
            tokenizer_name_or_path=tokenizer_name_or_path,
            allow_regex_split=not args.disable_state_regex_split,
        )
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以追加模式写出 JSONL。"""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_existing_problem_ids(path: Path) -> set[str]:
    """从已有输出 JSONL 中读取已完成的 problem_id。"""

    if not path.exists():
        return set()

    problem_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"输出 JSONL 第 {line_number} 行不是对象。")
            problem_id = record.get("problem_id")
            if isinstance(problem_id, str) and problem_id:
                problem_ids.add(problem_id)
    return problem_ids


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """写出 summary JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_run_summary(
    *,
    args: argparse.Namespace,
    summary_json: Path,
    step_max_new_tokens: int,
    rollout_max_new_tokens: int,
    state_segmenter: ReasoningStateSegmenter,
    input_problem_count: int,
    skipped_existing_count: int,
    newly_processed_count: int,
    output_problem_count: int,
    run_status: str,
) -> dict[str, Any]:
    """统一构建运行摘要，便于增量刷新与最终落盘共用。"""

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "summary_json": str(summary_json),
        "backend": args.backend,
        "model": args.model if args.backend == "vllm" else None,
        "input_problem_count": input_problem_count,
        "skipped_existing_count": skipped_existing_count,
        "newly_processed_count": newly_processed_count,
        "output_problem_count": output_problem_count,
        "run_status": run_status,
        "config": {
            "root_expansion_branches": args.root_expansion_branches,
            "expansion_branches": args.expansion_branches,
            "rollout_samples": args.rollout_samples,
            "num_simulations": args.num_simulations,
            "max_step_depth": args.max_step_depth,
            "step_max_new_tokens": step_max_new_tokens,
            "rollout_max_new_tokens": rollout_max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ucb_c": args.ucb_c,
            "store_rollouts": args.store_rollouts,
            "save_every": args.save_every,
            "resume": args.resume,
            "overwrite_output": args.overwrite_output,
            "state_boundary": {
                "max_step_chars": args.state_max_step_chars,
                "max_step_tokens": args.state_max_step_tokens,
                "regex_pattern": args.state_regex_pattern,
                "state_tokenizer": state_segmenter.config.tokenizer_name_or_path,
                "allow_regex_split": not args.disable_state_regex_split,
            },
        },
    }


async def run_self_check() -> int:
    """本地轻量自检。"""

    record = {
        "problem_id": "gsm8k-main-train-00000",
        "question": "What is 2 + 2?",
        "answer": "Compute carefully.\n#### 4",
        "final_answer": "4",
        "final_answer_numeric": "4",
        "prompt": "You are a careful math reasoning assistant.\nQuestion:\nWhat is 2 + 2?\n",
    }

    scripted_outputs = {
        "What is 2 + 2?\nStep 1: Compute carefully.": [
            "Step 2: 2 + 2 = 4.\n<answer>4</answer>",
            "Step 2: 2 + 2 = 4.\n<answer>4</answer>",
            "<answer>4</answer>",
            "<answer>4</answer>",
        ],
        "What is 2 + 2?\nGuess wildly.": ["<answer>5</answer>", "<answer>5</answer>"],
        "Question:\nWhat is 2 + 2?\n": [
            "Step 1: Compute carefully.\nStep 2: 2 + 2 = 4.\n<answer>4</answer>",
            "Guess wildly.\n<answer>5</answer>",
        ],
        "__default__": ["<answer>0</answer>"],
    }
    backend = MockGenerationBackend(scripted_outputs)
    state_segmenter = ReasoningStateSegmenter(StateBoundaryConfig(max_step_chars=128))
    engine = StepValueMCTSEngine(
        backend=backend,
        root_expansion_branches=2,
        expansion_branches=1,
        rollout_samples=2,
        max_step_depth=4,
        step_sampling_config=SamplingConfig(max_tokens=64),
        rollout_sampling_config=SamplingConfig(max_tokens=64),
        state_segmenter=state_segmenter,
        num_simulations=2,
        store_rollouts=True,
        ucb_c=1.2,
    )
    tree = await engine.build_problem_tree(record)
    await backend.aclose()

    q_values = [node["q_value"] for node in tree["nodes"] if node["q_value"] is not None]
    event_types = {event["event_type"] for event in tree["simulation_events"]}
    tmp_dir = Path(__file__).resolve().parent / ".tmp_build_mcts_self_check"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_output = tmp_dir / "trees.jsonl"
    try:
        append_jsonl(tmp_output, [tree])
        resumed_ids = load_existing_problem_ids(tmp_output)
    finally:
        if tmp_output.exists():
            tmp_output.unlink()
        if tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
    checks = {
        "has_multiple_nodes": tree["statistics"]["node_count"] >= 4,
        "has_correct_prefix": any(value == 1.0 for value in q_values),
        "has_wrong_prefix": any(value == 0.0 for value in q_values),
        "has_initial_paths": len(tree["initial_paths"]) == 2,
        "has_simulation_events": tree["statistics"]["simulation_count"] == 2,
        "has_expand_event": "expand_backup" in event_types,
        "has_visit_count": any(node["visit_count"] > 0 for node in tree["nodes"]),
        "has_terminal_node": any(node["is_terminal"] for node in tree["nodes"]),
        "resume_can_find_problem_id": record["problem_id"] in resumed_ids,
    }
    print(json.dumps(tree, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": checks}, ensure_ascii=False))
    passed = all(checks.values())
    print(f"self_check_passed={passed}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""

    parser = argparse.ArgumentParser(description="Build MCTS step-value data")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/gsm8k/core/train.jsonl"),
        help="核心实验集输入 JSONL。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/gsm8k/mcts/train_trees.jsonl"),
        help="树状价值数据输出 JSONL。",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="摘要 JSON 输出路径；不传则自动放到 output_jsonl 同目录。",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "vllm"],
        default="mock",
        help="生成后端。服务器实跑时请切到 vllm。",
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/Qwen/Qwen2.5-3B",
        help="vLLM 使用的模型路径或 Hugging Face repo id。",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="兼容旧版参数；未单独设置 step/rollout 时回退到它。")
    parser.add_argument("--step-max-new-tokens", type=int, help="单步扩展生成的最大 token 数。")
    parser.add_argument("--rollout-max-new-tokens", type=int, help="rollout 价值估计生成的最大 token 数。")
    parser.add_argument("--root-expansion-branches", type=int, default=2)
    parser.add_argument("--expansion-branches", type=int, default=2)
    parser.add_argument("--rollout-samples", type=int, default=3)
    parser.add_argument("--max-step-depth", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=8)
    parser.add_argument("--ucb-c", type=float, default=1.4)
    parser.add_argument("--state-max-step-chars", type=int, default=512)
    parser.add_argument("--state-max-step-tokens", type=int)
    parser.add_argument("--state-regex-pattern", default=r"(?=Step\s+\d+\s*:)")
    parser.add_argument("--state-tokenizer", help="状态切分 token 截断使用的 tokenizer。默认回退到 --model。")
    parser.add_argument("--disable-state-regex-split", action="store_true")
    parser.add_argument("--max-problems", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="每处理多少道题就增量写盘一次。")
    parser.add_argument("--resume", action="store_true", help="若输出 JSONL 已存在，则跳过其中已有的 problem_id。")
    parser.add_argument("--overwrite-output", action="store_true", help="非 resume 模式下允许覆盖已有输出文件。")
    parser.add_argument("--store-rollouts", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    """异步主入口。"""

    if args.self_check:
        return await run_self_check()

    if args.save_every <= 0:
        raise ValueError("--save-every 必须是正整数。")
    if args.resume and args.overwrite_output:
        raise ValueError("--resume 与 --overwrite-output 不能同时启用。")

    random.seed(args.seed)
    rows = iter_jsonl(args.input_jsonl)
    if args.max_problems > 0:
        rows = rows[: args.max_problems]
    input_problem_count = len(rows)

    backend = make_backend(args)
    state_segmenter = make_state_segmenter(args)
    step_max_new_tokens = args.step_max_new_tokens if args.step_max_new_tokens is not None else min(args.max_new_tokens, 128)
    rollout_max_new_tokens = (
        args.rollout_max_new_tokens if args.rollout_max_new_tokens is not None else args.max_new_tokens
    )
    summary_json = args.summary_json
    if summary_json is None:
        summary_json = args.output_jsonl.with_name(args.output_jsonl.stem + "_summary.json")

    existing_problem_ids: set[str] = set()
    if args.resume:
        existing_problem_ids = load_existing_problem_ids(args.output_jsonl)
    elif args.output_jsonl.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"输出文件已存在：{args.output_jsonl}。若要续跑请加 --resume；若要覆盖请加 --overwrite-output。"
            )
        args.output_jsonl.unlink()

    skipped_existing_count = 0
    if existing_problem_ids:
        filtered_rows: list[dict[str, Any]] = []
        for record in rows:
            problem_id = record.get("problem_id")
            if isinstance(problem_id, str) and problem_id in existing_problem_ids:
                skipped_existing_count += 1
                continue
            filtered_rows.append(record)
        rows = filtered_rows

    engine = StepValueMCTSEngine(
        backend=backend,
        root_expansion_branches=args.root_expansion_branches,
        expansion_branches=args.expansion_branches,
        rollout_samples=args.rollout_samples,
        max_step_depth=args.max_step_depth,
        step_sampling_config=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=step_max_new_tokens,
        ),
        rollout_sampling_config=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=rollout_max_new_tokens,
        ),
        state_segmenter=state_segmenter,
        num_simulations=args.num_simulations,
        store_rollouts=args.store_rollouts,
        ucb_c=args.ucb_c,
    )

    pending_rows: list[dict[str, Any]] = []
    newly_processed_count = 0
    output_problem_count = len(existing_problem_ids)
    try:
        for record in rows:
            tree = await engine.build_problem_tree(record)
            pending_rows.append(tree)
            newly_processed_count += 1

            if len(pending_rows) >= args.save_every:
                append_jsonl(args.output_jsonl, pending_rows)
                output_problem_count += len(pending_rows)
                pending_rows.clear()
                running_summary = build_run_summary(
                    args=args,
                    summary_json=summary_json,
                    step_max_new_tokens=step_max_new_tokens,
                    rollout_max_new_tokens=rollout_max_new_tokens,
                    state_segmenter=state_segmenter,
                    input_problem_count=input_problem_count,
                    skipped_existing_count=skipped_existing_count,
                    newly_processed_count=newly_processed_count,
                    output_problem_count=output_problem_count,
                    run_status="running",
                )
                write_summary_json(summary_json, running_summary)
    except Exception:
        if pending_rows:
            append_jsonl(args.output_jsonl, pending_rows)
            output_problem_count += len(pending_rows)
            pending_rows.clear()
        failed_summary = build_run_summary(
            args=args,
            summary_json=summary_json,
            step_max_new_tokens=step_max_new_tokens,
            rollout_max_new_tokens=rollout_max_new_tokens,
            state_segmenter=state_segmenter,
            input_problem_count=input_problem_count,
            skipped_existing_count=skipped_existing_count,
            newly_processed_count=newly_processed_count,
            output_problem_count=output_problem_count,
            run_status="failed",
        )
        write_summary_json(summary_json, failed_summary)
        raise
    finally:
        await backend.aclose()

    if pending_rows:
        append_jsonl(args.output_jsonl, pending_rows)
        output_problem_count += len(pending_rows)
        pending_rows.clear()

    summary = build_run_summary(
        args=args,
        summary_json=summary_json,
        step_max_new_tokens=step_max_new_tokens,
        rollout_max_new_tokens=rollout_max_new_tokens,
        state_segmenter=state_segmenter,
        input_problem_count=input_problem_count,
        skipped_existing_count=skipped_existing_count,
        newly_processed_count=newly_processed_count,
        output_problem_count=output_problem_count,
        run_status="completed",
    )
    write_summary_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """同步入口。"""

    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
