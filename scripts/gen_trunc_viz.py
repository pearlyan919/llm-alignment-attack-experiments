"""
实验三（步骤 2，纯 CPU 可视化）：ASR-截断比例折线图 + 逐样本热力图
输入：outputs/exp3/truncation_summary.json（truncation_test.py 产出）
输出：outputs/exp3/fig4_asr_truncation_curve.png
      outputs/exp3/fig5_truncation_heatmap.png
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

if not paths.setup_matplotlib_font():
    print("[提示] 未找到中文字体，图中中文可能显示异常", file=sys.stderr)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = paths.OUT_EXP3
os.makedirs(OUT, exist_ok=True)
d = json.load(open(os.path.join(OUT, "truncation_summary.json"), encoding="utf-8"))

RATIOS = d["ratios"]  # [0.1..1.0]
RATIO_KEYS = [str(int(r * 100)) for r in RATIOS]
asr = {r: d["asr_by_ratio"][str(r)]["asr"] * 100 for r in RATIOS}
multi = {r: d["asr_by_ratio"][str(r)]["num"] for r in RATIOS}

samples = d["samples"]
labels = ["%s\n(%s)" % (s["source"], s["goal"][:22]) for s in samples]

# ---------- 图1：ASR 截断曲线 ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = np.arange(10, 101, 10)
ax.plot(xs, [asr[r] for r in RATIOS], "o-", color="#d62728", lw=2.5,
        ms=8, label="单次判定 ASR")
# 标注 multi 采样 ASR（30/50/100 档）
for r in RATIOS:
    if r in (0.3, 0.5, 1.0):
        idx = int(r * 10) - 1
        m = d["asr_by_ratio"][str(r)]
        ax.plot(r * 100, m["asr"] * 100, "s", color="#1f77b4", ms=10)
        ax.annotate("5x采样\n%.0f%%" % (m["asr"] * 100),
                    (r * 100, m["asr"] * 100), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#1f77b4",
                    fontweight="bold")
ax.axvline(50, ls="--", color="gray", lw=1, alpha=0.7)
ax.axvline(30, ls="--", color="gray", lw=1, alpha=0.7)
ax.text(50, ax.get_ylim()[1], " 50%", color="gray", fontsize=10)
ax.text(30, ax.get_ylim()[1], " 30%", color="gray", fontsize=10)
ax.set_xlabel("保留后缀前 k% 的 Token", fontsize=12)
ax.set_ylabel("Gemma-2 越狱成功率 ASR (%)", fontsize=12)
ax.set_title("实验三：截断测试 — ASR 随后缀保留比例的变化（5 个迁移后缀）",
             fontsize=13, fontweight="bold")
ax.set_xticks(xs)
ax.set_ylim(-5, 105)
ax.grid(alpha=0.3)
ax.legend(loc="center right", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_asr_truncation_curve.png"), dpi=150)
plt.close(fig)

# ---------- 图2：逐样本热力图（确定性判定） ----------
M = np.array([[1 if s["ratios"][k]["jailbroken"] else 0 for k in RATIO_KEYS]
              for s in samples])
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(M, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(RATIO_KEYS)))
ax.set_xticklabels(["%s%%" % k for k in RATIO_KEYS])
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(samples)):
    for j in range(len(RATIO_KEYS)):
        v = M[i, j]
        color = "white" if v == 1 else "black"
        ax.text(j, i, "越狱" if v else "拒绝", ha="center", va="center",
                color=color, fontsize=8)
ax.set_xlabel("保留比例（前缀 Token 数占比）", fontsize=11)
ax.set_title("实验三：逐样本截断越狱结果（■=越狱成功 ■=拒绝）", fontsize=13,
             fontweight="bold")
cb = fig.colorbar(im, ax=ax, ticks=[0, 1], fraction=0.03, pad=0.02)
cb.ax.set_yticklabels(["拒绝", "越狱"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_truncation_heatmap.png"), dpi=150)
plt.close(fig)

print("已生成 fig4_asr_truncation_curve.png / fig5_truncation_heatmap.png @ %s" % OUT)

