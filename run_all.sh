#!/usr/bin/env bash
# =============================================================================
# 一键复现：从 results/*.json（原始 GCG 结果）与 outputs/* 中间产物，
# 重新生成三个实验的全部图表 / CSV / 派生 JSON（纯 CPU，无需 GPU 与模型）。
#
#   bash run_all.sh
#
# 若 outputs/exp2、outputs/exp3 中的 GPU 推理产物（transfer_records.csv /
# truncation_records.csv）缺失，说明该仓库未包含实测结果，请先运行
# run_full_pipeline.sh（需要 GPU + 模型）再执行本脚本。
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

echo "======================================================================"
echo "[阶段 1/3] 实验一：Llama-2-7B vs Llama-3.2-3B GCG 攻击对比（纯 CPU）"
echo "======================================================================"
$PY scripts/generate_compare.py
$PY scripts/compare_llama2_llama3.py

echo "======================================================================"
echo "[阶段 2/3] 实验二：对抗后缀跨模型迁移攻击"
echo "======================================================================"
# 从三个权威 GCG 结果提取最终后缀（无需 GPU）
$PY scripts/extract_suffixes.py

if [ -f outputs/exp2/transfer_records.csv ]; then
    echo ">> 检测到已归档迁移测试记录，跳过 GPU 推理（transfer_test.py）"
else
    echo "!! 缺少 outputs/exp2/transfer_records.csv（需 GPU+模型生成）"
    echo "   请先运行 bash run_full_pipeline.sh，或按 README 手动推理后重跑。"
fi
if [ -f outputs/exp2/transfer_matrix.json ]; then
    $PY scripts/gen_heatmap.py
fi
if [ -f outputs/exp2/transfer_records.csv ]; then
    $PY scripts/common_matrix.py
else
    echo ">> 跳过 common_matrix.py（依赖迁移测试记录）"
fi

echo "======================================================================"
echo "[阶段 3/3] 实验三：后缀语义指纹与截断测试"
echo "======================================================================"
if [ -f outputs/exp3/truncation_records.csv ]; then
    echo ">> 检测到已归档截断测试记录，跳过 GPU 推理（truncation_test.py）"
else
    echo "!! 缺少 outputs/exp3/truncation_records.csv（需 GPU+模型生成）"
    echo "   请先运行 bash run_full_pipeline.sh，或按 README 手动推理后重跑。"
fi
if [ -f outputs/exp3/truncation_summary.json ]; then
    $PY scripts/gen_trunc_viz.py
fi
if [ -f outputs/exp3/truncation_records.csv ]; then
    $PY scripts/gen_trunc_strict.py
else
    echo ">> 跳过 gen_trunc_strict.py（依赖截断测试记录）"
fi

echo "======================================================================"
echo "复现完成。产物位置："
echo "  实验一：outputs/exp1/   (fig1_dual_axis_compare.png / fig2_per_goal_loss.png"
echo "                          / llama2_vs_llama3_loss_asr.png / llama2_vs_llama3_metrics.csv)"
echo "  实验二：outputs/exp2/   (suffixes.json / fig3_transfer_heatmap.png ...)"
echo "  实验三：outputs/exp3/   (fig4_*.png / fig5_*.png / truncation_strict.json ...)"
echo "详细结论见 reports/ 下的三份实验报告。"
echo "======================================================================"
