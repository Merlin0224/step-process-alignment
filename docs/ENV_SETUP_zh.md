# 阶段 1 环境配置说明

这套配置面向 `autodl + RTX 3090 (24GB) + pixi`，目标是先把后续 `Step-DPO / Step-GRPO / MCTS` 会用到的依赖框架铺平。

本项目的 `pixi.toml` 已显式把主 PyPI 索引固定为 `https://pypi.org/simple`。这样可以绕开部分镜像站对 `*.whl.metadata` 支持不完整导致的 `404` 问题。

## 为什么拆成两个 pixi 环境

不建议把 `LLaMA-Factory`、`veRL/vLLM`、`TRL` 强行塞进一个 Python 环境里，原因是：

1. `LLaMA-Factory v0.9.4` 官方已经切到 `Python 3.11-3.13`。
2. `veRL + vLLM 0.8.x` 官方稳定文档仍明显偏向 `Python 3.10`。
3. 你当前项目后续本来就是两条链路并行：
   - `dpo` 环境负责 `SFT / DPO / LLaMA-Factory`
   - `grpo` 环境负责 `TRL / vLLM / 后续 veRL 改造`

所以这里采用的是“同一个 pixi 工作区，两个独立环境”的管理方式。

## 文件说明

- `pixi.toml`
  - `dpo` 环境：`Python 3.11 + torch 2.6 + LLaMA-Factory 0.9.4`
  - `grpo` 环境：`Python 3.10 + torch 2.6 + vLLM 0.8.3 + TRL`
- `scripts/check_stack.py`
  - 用于在 autodl 上安装完后快速验证关键包和 CUDA 是否正常。

## 在 autodl 上的推荐步骤

### 1. 安装 pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
pixi --version
```

### 2. 进入项目目录

```bash
cd /root/autodl-tmp/your-project
```

把这里仓库中的 `pixi.toml`、`scripts/`、`docs/` 同步过去即可。

### 3. 先检查驱动

```bash
nvidia-smi
```

建议确认驱动足够新，能稳定跑 `CUDA 12.4` 对应的 PyTorch wheel。当前 `pixi.toml` 里使用的是：

```toml
extra-index-urls = ["https://download.pytorch.org/whl/cu124"]
```

如果你租到的 autodl 镜像驱动偏老，后面装包时报 CUDA 轮子不兼容，可以把它改成 `cu121` 再重装。

### 4. 安装 DPO 环境

```bash
pixi install -e dpo
pixi run -e dpo verify
```

### 5. 安装 GRPO 环境

```bash
pixi install -e grpo
pixi run -e grpo verify
```

## 关于 veRL

这份 `pixi.toml` 先把 `Step-GRPO` 需要的底座装好：`torch + trl + vllm + ray + tensordict`。

`veRL` 我建议等你真正进入阶段 4 再按官方源码方式安装，而不是一开始就锁死进求解器，原因是：

1. `veRL` 的推荐安装组合和 `vLLM`、`ray`、`flash-attn` 耦合很强。
2. 后续你大概率要直接修改 `GRPOTrainer` 或相邻代码，源码安装更方便。
3. 用 pixi 先把基础 CUDA/Python 依赖打平，后面再 `pip install -e` veRL，排障成本更低。

推荐到阶段 4 时再执行：

```bash
pixi shell -e grpo
git clone https://github.com/volcengine/verl.git external/verl
python -m pip install --no-deps -e external/verl
```

如果后面你明确决定“只用 TRL，不碰 veRL”，那这个 `grpo` 环境已经够你开始手改 `GRPOTrainer` 了。

## 关于 flash-attn

我没有把 `flash-attn` 直接写进 `pixi` 的默认求解依赖里，原因是它在新环境冷启动时最容易因为编译链、CUDA 头文件或 `torch` 尚未就位而失败。

更稳妥的方式是：

1. 先把 `pixi install -e dpo` 或 `pixi install -e grpo` 跑通
2. 确认 `pixi run -e ... verify` 没问题
3. 再进入对应环境手动安装 `flash-attn`

示例：

```bash
pixi shell -e dpo
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 3090 上的额外建议

### DPO / SFT

- 优先 `LoRA` 或 `QLoRA`
- 打开 `gradient_checkpointing`
- 优先 `bf16`
- 单卡先以 `Qwen2.5-1.5B` 跑通，再切 `3B`

### vLLM / MCTS

- 先用较小的 `max_model_len`
- 树搜索 rollout 分支数 `K` 从 `2` 或 `3` 开始
- 明确开启 prefix caching
- 生成和训练尽量分卡，避免同卡争抢显存

## 目前这套版本选择的依据

这次我按官方资料做了保守选型：

1. PyTorch 官方提供 `torch==2.6.0` 对应的 `cu124` wheel。
2. `LLaMA-Factory` 官方文档推荐 `torch 2.6.0`，且 `v0.9.4` 已要求 `Python 3.11-3.13`。
3. `veRL` 官方文档与升级说明里，对 `vLLM 0.8.2 / 0.8.3`、`torch 2.6.0` 的组合给出了明确示例。

## 下一步建议

环境装好后，我们就可以马上进入阶段 1 的第二项：

1. 写 `Rule-based Verifier`
2. 约定统一输出格式，例如 `<answer>...</answer>`
3. 先在 `GSM8K` 上把答案抽取和正确率判断跑通
