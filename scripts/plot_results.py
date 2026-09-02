"""
统一画图入口：一键生成三个实验的全部图表。
调用现有的各 gen_*.py / generate_compare.py / compare_llama2_llama3.py，
并将代表性图表汇总到 figures/ 目录。
"""
import os, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "scripts/generate_compare.py",
    "scripts/compare_llama2_llama3.py",
    "scripts/gen_heatmap.py",
    "scripts/gen_trunc_viz.py",
]

print("=" * 70)
print("[plot_results] 开始统一生成全部图表")
print("=" * 70)

for script in SCRIPTS:
    path = ROOT / script
    if not path.exists():
        print(f"[跳过] 未找到 {script}")
        continue
    print(f"\n>> 执行 {script}")
    ret = os.system(f"cd {ROOT} && python {path}")
    if ret != 0:
        print(f"[警告] {script} 返回非零码 {ret}")

# 汇总代表性图表到 figures/
copy_map = {
    ROOT / "outputs" / "exp1" / "llama2_vs_llama3_loss_asr.png": FIGURES / "loss_asr.png",
    ROOT / "outputs" / "exp2" / "fig3_transfer_heatmap.png": FIGURES / "transfer_heatmap.png",
}

print("\n[汇总] 将代表性图表复制到 figures/")
for src, dst in copy_map.items():
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  {src.name} -> figures/{dst.name}")
    else:
        print(f"  [缺失] {src}")

print("\n[完成] 全部图表已生成，详见 outputs/exp1~3/ 与 figures/")

