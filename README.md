# LLM Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)" by [Andy Zou](https://andyzoujm.github.io/), [Zifan Wang](https://sites.google.com/west.cmu.edu/zifan-wang/home), [Nicholas Carlini](https://nicholas.carlini.com/), [Milad Nasr](https://people.cs.umass.edu/~milad/), [J. Zico Kolter](https://zicokolter.com/), and [Matt Fredrikson](https://www.cs.cmu.edu/~mfredrik/).

Check out our [website and demo here](https://llm-attacks.org/).

## Updates
- (2024-08-01) We release `nanogcg`, a fast and easy-to-use implementation of the GCG algorithm. `nanogcg` can be installed via pip and the code is available [here](https://github.com/GraySwanAI/nanoGCG/tree/main).
- (2023-08-16) We include a notebook `demo.ipynb` (or see it on [Colab](https://colab.research.google.com/drive/1dinZSyP1E4KokSLPcCh1JQFUFsN-WV--?usp=sharing)) containing the minimal implementation of GCG for jailbreaking LLaMA-2 for generating harmful completion.


## Table of Contents

- [Supplementary Experiments (this repo)](#supplementary-experiments-this-repo)
- [Installation](#installation)
- [Models](#models)
- [Experiments](#experiments)
- [Demo](#demo)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Citation](#citation)

## Supplementary Experiments (this repo)

本仓库在官方 GCG 攻击引擎之上，扩展了三组**对齐防御对比实验**（中文报告见 `reports/`），
并整理了可直接复现的全部中间产物与图表（`outputs/exp1 ~ exp3`）、原始攻击结果（`results/`）。

| 实验 | 对比对象 | 目标 | 关键结论（详见 `reports/`） |
|---|---|---|---|
| 实验一 | Llama-2-7B-Chat vs Llama-3.2-3B-Instruct | AdvBench 前 10 条 harmful behaviors | 500 步 GCG 后 Llama-2 仅 1/10 被攻破，Llama-3.2 达 10/10，防御能力显著分化 |
| 实验二 | 后缀在 3 个模型间的迁移 | 10+7 条行为（异构迁移 + 行为集扩展） | Gemma-2 后缀→Llama 系完全迁移；Llama 系→Gemma-2 仅 ~30% |
| 实验三 | 后缀"语义指纹"截断 | 5 个迁移成功后缀，token 级保留 10%~100% | 越狱依赖后缀"关键头部"；30%~60% 截断下 ASR 并不单调，须逐样本细看 |

### 一键复现（纯 CPU，无需 GPU 与模型）

`results/*.json` 已含全部权威攻击结果，`outputs/exp2`、`outputs/exp3` 已含 GPU 推理产物，
因此可**不加载任何模型**地重新生成全部图表 / CSV / 派生 JSON：

```bash
pip install -r requirements-reproduce.txt
bash run_all.sh          # Linux / WSL / macOS
run_all.bat              # Windows（双击或 cmd 执行）
```

复现产物：
- `outputs/exp1/`：`fig1_dual_axis_compare.png`、`fig2_per_goal_loss.png`、
  `llama2_vs_llama3_loss_asr.png`、`llama2_vs_llama3_metrics.csv`
- `outputs/exp2/`：`suffixes.json`、`fig3_transfer_heatmap.png`、`transfer_matrix.json`、`transfer_records.csv`
- `outputs/exp3/`：`truncation_summary.json`、`truncation_records.csv`、`fig4_asr_truncation_curve.png`、
  `fig5_truncation_heatmap.png`、`truncation_strict.json`

### 完整管线（GPU + 模型，重新执行攻击与推理）

```bash
pip install -e . && pip install -r requirements-gpu.txt   # 并按需安装 CUDA 版 torch
# 模型放到 models/（或设置 GCG_MODELS_DIR）：
#   models/llama-2-7b-chat-hf  models/Llama-3.2-3B-Instruct  models/gemma-2-2b-it
bash run_full_pipeline.sh
```

管线阶段：`experiments/main.py` 对三模型执行 GCG 攻击（配置见
`experiments/configs/individual_{llama2,llama3.2,gemma2}.py`，参数已与归档结果一致）→
`scripts/extract_suffixes.py` 提取后缀 → `scripts/transfer_test.py` / `scripts/truncation_test.py`
推理 → 各 `scripts/gen_*.py` 绘图。

### 脚本清单（scripts/）

| 脚本 | 阶段 | 输入 → 输出 |
|---|---|---|
| `extract_suffixes.py` | 纯 CPU | `results/*.json` → `outputs/exp2/suffixes.json` |
| `generate_compare.py` / `compare_llama2_llama3.py` | 纯 CPU | `results/*.json` → 实验一图与 CSV |
| `transfer_test.py` | GPU | `suffixes.json` → `transfer_records.csv` + `transfer_matrix.json` |
| `common_matrix.py` | 纯 CPU | 上述产物 → 共同目标迁移矩阵（打印） |
| `gen_heatmap.py` | 纯 CPU | `transfer_matrix.json` → `fig3_transfer_heatmap.png` |
| `truncation_test.py` | GPU | `suffixes.json` → `truncation_records.csv` + `truncation_summary.json` |
| `gen_trunc_viz.py` / `gen_trunc_strict.py` | 纯 CPU | 上述产物 → `fig4/fig5` + `truncation_strict.json` |

> `scripts/paths.py` 统一管理仓库根/结果/输出目录：模型目录可用 `GCG_MODELS_DIR`
> 覆盖；所有脚本不再依赖原始开发机的绝对路径。
> `scratch/` 保留开发过程中的临时脚本与 FastChat 模板补丁示例，仅供追溯，不在复现路径内。

## Installation

We need the newest version of FastChat `fschat==0.2.23` and please make sure to install this version. The `llm-attacks` package can be installed by running the following command at the root of this repository:

```bash
pip install -e .
```

## Models

Please follow the instructions to download Vicuna-7B or/and LLaMA-2-7B-Chat first (we use the weights converted by HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)).  Our script by default assumes models are stored in a root directory named as `/DIR`. To modify the paths to your models and tokenizers, please add the following lines in `experiments/configs/individual_xxx.py` (for individual experiment) and `experiments/configs/transfer_xxx.py` (for multiple behaviors or transfer experiment). An example is given as follows.

```python
    config.model_paths = [
        "/DIR/vicuna/vicuna-7b-v1.3",
        ... # more models
    ]
    config.tokenizer_paths = [
        "/DIR/vicuna/vicuna-7b-v1.3",
        ... # more tokenizers
    ]
```

## Demo
We include a notebook `demo.ipynb` which provides an example on attacking LLaMA-2 with GCG. You can also view this notebook on [Colab](https://colab.research.google.com/drive/1dinZSyP1E4KokSLPcCh1JQFUFsN-WV--?usp=sharing). This notebook uses a minimal implementation of GCG so it should be only used to get familiar with the attack algorithm. For running experiments with more behaviors, please check Section Experiments. To monitor the loss in the demo we use `livelossplot`, so one should install this library first by pip.

```bash
pip install livelossplot
```

## Experiments 

The `experiments` folder contains code to reproduce GCG experiments on AdvBench.

- To run individual experiments with harmful behaviors and harmful strings (i.e. 1 behavior, 1 model or 1 string, 1 model), run the following code inside `experiments` (changing `vicuna` to `llama2` and changing `behaviors` to `strings` will switch to different experiment setups):

```bash
cd launch_scripts
bash run_gcg_individual.sh vicuna behaviors
```

- To perform multiple behaviors experiments (i.e. 25 behaviors, 1 model), run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gcg_multiple.sh vicuna # or llama2
```

- To perform transfer experiments (i.e. 25 behaviors, 2 models), run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gcg_transfer.sh vicuna 2 # or vicuna_guanaco 4
```

- To perform evaluation experiments, please follow the directions in `experiments/parse_results.ipynb`.

Notice that all hyper-parameters in our experiments are handled by the `ml_collections` package [here](https://github.com/google/ml_collections). You can directly change those hyper-parameters at the place they are defined, e.g. `experiments/configs/individual_xxx.py`. However, a recommended way of passing different hyper-parameters -- for instance you would like to try another model -- is to do it in the launch script. Check out our launch scripts in `experiments/launch_scripts` for examples. For more information about `ml_collections`, please refer to their [repository](https://github.com/google/ml_collections).

## Reproducibility

A note for hardware: all experiments we run use one or multiple NVIDIA A100 GPUs, which have 80G memory per chip. 

We include a few examples people told us when reproducing our results. They might also include workaround for solving a similar issue in your situation. 

- [Prompting Llama-2-7B-Chat-GGML](https://github.com/llm-attacks/llm-attacks/issues/8)
- [Possible Naming Issue for Running Experiments on Windows](https://github.com/llm-attacks/llm-attacks/issues/28)

Currently the codebase only supports training with LLaMA or Pythia based models. Running the scripts with other models (with different tokenizers) will likely result in silent errors. As a tip, start by modifying [this function](https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/base/attack_manager.py#L130) where different slices are defined for the model.

## License
`llm-attacks` is licensed under the terms of the MIT license. See LICENSE for more details.


