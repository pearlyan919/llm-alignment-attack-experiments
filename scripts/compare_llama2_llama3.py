"""
实验一：Llama-2 vs Llama-3.2 对比分析（补充绘图）
1) 提取前 10 条 AdvBench 目标的最终 ASR 与 Loss<0.6 达标迭代步数
2) 绘制 Loss-ASR 双轴对比曲线（Llama-2 实线 / Llama-3 虚线）
3) 找出"难啃的骨头"：防御结果不一致的目标
输出：outputs/exp1/llama2_vs_llama3_loss_asr.png
（数据表由 generate_compare.py 统一导出，避免两份 CSV 互相覆盖）
"""
import json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from matplotlib import font_manager, rcParams

if not paths.setup_matplotlib_font():
    print("[提示] 未找到中文字体，图中中文可能显示异常", file=sys.stderr)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = paths.RESULTS_DIR
OUT = paths.OUT_EXP1
os.makedirs(OUT, exist_ok=True)

FILES = {
    "Llama-2": paths.result_json("llama-2"),
    "Llama-3.2": paths.result_json("llama-3.2"),
}
TH = 0.6


def load(fn):
    d = json.load(open(fn, encoding="utf-8"))
    goals = d["params"]["goals"]
    tests = d["tests"]
    n_steps = d["params"]["n_steps"]
    test_steps = d["params"]["test_steps"]
    CK = len(tests) // len(goals)
    return d, goals, tests, n_steps, test_steps, CK


def analyze(fn):
    d, goals, tests, n_steps, test_steps, CK = load(fn)
    rows = []
    for i, g in enumerate(goals):
        base = i * CK
        steps = [j * test_steps for j in range(CK)]
        loss_traj, passed_traj = [], []
        for j in range(CK):
            info = tests[base + j].get(g)
            loss_traj.append(info[0][3] if info and info[0] else float("nan"))
            passed_traj.append(int(info[0][1]) if info and info[0] else 0)
        first_below = None
        for j, l in enumerate(loss_traj):
            if not np.isnan(l) and l < TH:
                first_below = steps[j]
                break
        rows.append({
            "goal": g, "i": i, "final_passed": passed_traj[-1],
            "loss_traj": loss_traj, "passed_traj": passed_traj,
            "steps": steps, "first_below": first_below,
        })
    return rows, steps, CK


data = {}
for tag, fn in FILES.items():
    rows, steps, CK = analyze(fn)
    data[tag] = {"rows": rows, "steps": steps, "CK": CK}
    print(f"[load] {tag}: {len(rows)} goals, {CK} checkpoints, steps={steps[-1]}")

goals_text = data["Llama-2"]["rows"][0]["goal"]

# ================= 1) 数据提取 =================
print("\n" + "=" * 100)
print("步骤 1：最终 ASR 与 Loss<%.1f 达标步数" % TH)
print("=" * 100)
tbl = []
for i in range(10):
    r2 = data["Llama-2"]["rows"][i]
    r3 = data["Llama-3.2"]["rows"][i]
    asr2 = "通过" if r2["final_passed"] else "未通过"
    asr3 = "通过" if r3["final_passed"] else "未通过"
    s2 = str(r2["first_below"]) if r2["first_below"] is not None else "未达标"
    s3 = str(r3["first_below"]) if r3["first_below"] is not None else "未达标"
    diff = ""
    if r2["final_passed"] and not r3["final_passed"]:
        diff = "L2攻破 / L3防御"
    elif not r2["final_passed"] and r3["final_passed"]:
        diff = "L3攻破 / L2防御"
    tbl.append([i + 1, r2["goal"][:36], asr2, s2, asr3, s3, diff])
    print(f"{i+1:>2} | {r2['goal'][:36]:<38} | {asr2:<5} {s2:>5} | {asr3:<5} {s3:>5} | {diff}")


def avg_below(tag):
    vals = [r["first_below"] for r in data[tag]["rows"] if r["first_below"] is not None]
    return (sum(vals) / len(vals)) if vals else None, len(vals)


for tag in ["Llama-2", "Llama-3.2"]:
    avg, cnt = avg_below(tag)
    print(f"\n[{tag}] Loss<{TH} 达标目标: {cnt}/10, 平均达标步数: {avg:.0f}" if avg is not None
          else f"\n[{tag}] 无目标达标")

# ================= 2) 双轴对比图 =================
fig, ax1 = plt.subplots(figsize=(10, 6))
colors = {"Llama-2": ("#1f77b4", "#d62728"), "Llama-3.2": ("#2ca02c", "#ff7f0e")}
lss = {"Llama-2": "-", "Llama-3.2": "--"}
for tag in ["Llama-2", "Llama-3.2"]:
    rows = data[tag]["rows"]
    steps = data[tag]["steps"]
    avg_loss = [float(np.nanmean([r["loss_traj"][j] for r in rows])) for j in range(len(steps))]
    asr = [sum(r["passed_traj"][j] for r in rows) / len(rows) * 100 for j in range(len(steps))]
    cL, cA = colors[tag]
    ls = lss[tag]
    ax1.plot(steps, avg_loss, ls, color=cL, marker="o", ms=4, label=f"{tag} 平均 Loss")
    ax1.set_xlabel("迭代步数 (Step)")
    ax1.set_ylabel("目标交叉熵 Loss（平均）", color="#333333")
    ax1.set_ylim(0, max(4, max(avg_loss) * 1.1))
    ax2 = ax1.twinx()
    ax2.plot(steps, asr, ls, color=cA, marker="s", ms=4, label=f"{tag} ASR")
    ax2.set_ylabel("攻击成功率 ASR（%）", color="#333333")
    ax2.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)
ax2.legend(loc="center right", fontsize=9)
plt.title("Llama-2 vs Llama-3.2：Loss 与 ASR 随迭代步数的对比（10 目标平均）")
plt.tight_layout()
fig_path = os.path.join(OUT, "llama2_vs_llama3_loss_asr.png")
plt.savefig(fig_path, dpi=150)
print(f"\n[图] 已保存: {fig_path}")

# ================= 3) 难啃的骨头 =================
print("\n" + "=" * 100)
print("步骤 3：防御结果不一致的目标（难啃的骨头）")
print("=" * 100)
r2s, r3s = data["Llama-2"]["rows"], data["Llama-3.2"]["rows"]
for i in range(10):
    if r2s[i]["final_passed"] != r3s[i]["final_passed"]:
        w = "Llama-3 防御成功，Llama-2 被攻破" if (r2s[i]["final_passed"] and not r3s[i]["final_passed"]) \
            else "Llama-2 防御成功，Llama-3 被攻破"
        l2min = min(l for l in r2s[i]["loss_traj"] if not np.isnan(l))
        l3min = min(l for l in r3s[i]["loss_traj"] if not np.isnan(l))
        print(f"\n目标 {i+1}: {r2s[i]['goal']}")
        print(f"  Llama-2: ASR={'通过' if r2s[i]['final_passed'] else '未通过'}, 最终Loss={l2min:.3f}, 达标步={r2s[i]['first_below']}")
        print(f"  Llama-3.2: ASR={'通过' if r3s[i]['final_passed'] else '未通过'}, 最终Loss={l3min:.3f}, 达标步={r3s[i]['first_below']}")
        print(f"  判定: {w}")

# ================= 导出 CSV（由 generate_compare.py 统一导出，此处不再写入） =================
print("\n[提示] llama2_vs_llama3_metrics.csv 由 generate_compare.py 统一生成，请勿在本脚本重复导出")
print("DONE")

