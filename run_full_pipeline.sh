#!/usr/bin/env bash
# =============================================================================
# 完整管线：重新执行 GCG 攻击 -> 后缀提取 -> 跨模型迁移/截断推理 -> 可视化
#
# 前置条件：
#   1) GPU 环境，已按 README「安装与依赖」安装 torch / fastchat / transformers
#      / bitsandbytes 与 llm-attacks（pip install -e .）
#   2) 模型权重位于 models/（或设置 GCG_MODELS_DIR 指向外部目录）：
#        models/llama-2-7b-chat-hf      （Llama-2-7B-Chat）
#        models/Llama-3.2-3B-Instruct   （Llama-3.2-3B-Instruct）
#        models/gemma-2-2b-it           （Gemma-2-2B-IT）
#   3) 数据：data/advbench/harmful_behaviors.csv（实验一/二的前 10 条 + 实验二
#      的迁移行为集见 data/transfer_expriment_behaviors.csv）
#
# 用法：
#   bash run_full_pipeline.sh
#
# 说明：攻击运行数千步耗时较长，建议在 tmux/nohup 下运行。
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

echo "== [0] 依赖与模型检查 =="
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
$PY -c "import llm_attacks, fastchat; print('llm_attacks OK / fastchat OK')"
MODELS_DIR="${GCG_MODELS_DIR:-$ROOT/models}"
for sub in llama-2-7b-chat-hf Llama-3.2-3B-Instruct gemma-2-2b-it; do
    if [ ! -d "$MODELS_DIR/$sub" ]; then
        echo "!! 缺少模型目录: $MODELS_DIR/$sub"
        echo "   请下载权重后重试，或设置 GCG_MODELS_DIR 指向模型根目录。"
        exit 1
    fi
done
echo ">> 模型目录 OK: $MODELS_DIR"

echo "======================================================================"
echo "[1/6] GCG 攻击：三个目标模型（结果写入 results/*.json）"
echo "======================================================================"
cd experiments
for cfg in individual_llama2 individual_llama3.2 individual_gemma2; do
    echo "---- 运行 config: $cfg ----"
    $PY -u main.py --config "configs/${cfg}.py"
done
cd "$ROOT"

echo "======================================================================"
echo "[2/6] 提取三个模型的最终攻击后缀"
echo "======================================================================"
$PY scripts/extract_suffixes.py

echo "======================================================================"
echo "[3/6] 跨模型迁移测试（GPU，需 3 个模型）"
echo "======================================================================"
$PY scripts/transfer_test.py

echo "======================================================================"
echo "[4/6] 截断测试：后缀语义指纹（GPU，Gemma-2）"
echo "======================================================================"
$PY scripts/truncation_test.py

echo "======================================================================"
echo "[5/6] 可视化：图 3/4/5 与派生 JSON"
echo "======================================================================"
$PY scripts/gen_heatmap.py
$PY scripts/common_matrix.py
$PY scripts/gen_trunc_viz.py
$PY scripts/gen_trunc_strict.py

echo "======================================================================"
echo "[6/6] 一键复现其余图表/CSV（纯 CPU）"
echo "======================================================================"
$PY scripts/generate_compare.py
$PY scripts/compare_llama2_llama3.py

echo "完整管线结束。产物见 outputs/exp1 ~ exp3，报告见 reports/。"
