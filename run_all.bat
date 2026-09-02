@echo off
REM ===========================================================================
REM 一键复现（Windows / 纯 CPU）：从 results\*.json 与 outputs\* 中间产物
REM 重新生成三个实验的全部图表 / CSV / 派生 JSON。
REM
REM 用法：双击本文件，或在该目录打开终端后执行  run_all.bat
REM 依赖：python（>=3.9）+ numpy + matplotlib，详见 requirements-reproduce.txt
REM ===========================================================================
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ======================================================================
echo [1/3] 实验一：Llama-2-7B vs Llama-3.2-3B GCG 攻击对比
echo ======================================================================
python scripts\generate_compare.py || goto :fail
python scripts\compare_llama2_llama3.py || goto :fail

echo ======================================================================
echo [2/3] 实验二：对抗后缀跨模型迁移攻击
echo ======================================================================
python scripts\extract_suffixes.py || goto :fail
if exist outputs\exp2\transfer_records.csv (
    echo [信息] 已归档迁移测试记录存在，跳过 GPU 推理
) else (
    echo [警告] 缺少 outputs\exp2\transfer_records.csv，需 GPU+模型生成
    echo        请先运行 run_full_pipeline.sh（Linux/WSL）后重跑本脚本
)
if exist outputs\exp2\transfer_matrix.json (
    python scripts\gen_heatmap.py || goto :fail
)
if exist outputs\exp2\transfer_records.csv (
    python scripts\common_matrix.py || goto :fail
) else (
    echo [信息] 跳过 common_matrix.py（依赖迁移测试记录）
)

echo ======================================================================
echo [3/3] 实验三：后缀语义指纹与截断测试
echo ======================================================================
if exist outputs\exp3\truncation_records.csv (
    echo [信息] 已归档截断测试记录存在，跳过 GPU 推理
) else (
    echo [警告] 缺少 outputs\exp3\truncation_records.csv，需 GPU+模型生成
)
if exist outputs\exp3\truncation_summary.json (
    python scripts\gen_trunc_viz.py || goto :fail
)
if exist outputs\exp3\truncation_records.csv (
    python scripts\gen_trunc_strict.py || goto :fail
) else (
    echo [信息] 跳过 gen_trunc_strict.py（依赖截断测试记录）
)

echo ======================================================================
echo 复现完成。产物见 outputs\exp1 ~ exp3，报告见 reports\。
echo ======================================================================
endlocal
exit /b 0

:fail
echo.
echo [错误] 复现过程中出现异常，请检查上方报错信息。
endlocal
exit /b 1
