"""
生成 3x3 跨模型迁移攻击热力图
输入：outputs/exp2/transfer_matrix.json（transfer_test.py 产出）
输出：outputs/exp2/fig3_transfer_heatmap.png
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from matplotlib import font_manager, rcParams

if not paths.setup_matplotlib_font():
    print("[提示] 未找到中文字体，图中中文可能显示异常", file=sys.stderr)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = paths.OUT_EXP2
m = json.load(open(os.path.join(OUT, "transfer_matrix.json"), encoding="utf-8"))
MAT = m["matrix"]

SOURCES = ["llama-2", "llama-3.2", "gemma-2"]
TARGETS = ["llama-2", "llama-3.2", "gemma-2"]
LABELS = {"llama-2": "Llama-2\n7B-Chat", "llama-3.2": "Llama-3.2\n3B-Instruct",
          "gemma-2": "Gemma-2\n2B-IT"}
FAMILY = {"llama-2": "Llama", "llama-3.2": "Llama", "gemma-2": "Gemma"}

data = np.zeros((3, 3))
ann = [[""] * 3 for _ in range(3)]
for i, s in enumerate(SOURCES):
    for j, t in enumerate(TARGETS):
        v = MAT["%s->%s" % (s, t)]
        pct = 100.0 * v["num"] / v["denom"]
        data[i, j] = pct
        ann[i][j] = "%d/%d\n%.0f%%" % (v["num"], v["denom"], pct)

fig, ax = plt.subplots(figsize=(9.5, 7.5))
cmap = plt.cm.Reds
norm = plt.Normalize(0, 100)
for i in range(3):
    for j in range(3):
        v = data[i, j]
        color = cmap(norm(v))
        edge = "#1a3a5c" if i == j else "black"
        ax.add_patch(plt.Rectangle((j, 3 - 1 - i), 1, 1, fill=True, facecolor=color,
                                   edgecolor=edge, lw=2.5 if i == j else 1))
        ax.text(j + 0.5, 3 - 1 - i + 0.33, ann[i][j], ha="center", va="center",
                fontsize=17, fontweight="bold", color="#333333")
        tag = ("基准" if i == j else
               ("同构迁移" if FAMILY[SOURCES[i]] == FAMILY[TARGETS[j]] else "异构迁移"))
        ax.text(j + 0.5, 3 - 1 - i - 0.12, tag, ha="center", va="center",
                fontsize=10, color="#666666")

ax.set_xlim(0, 3); ax.set_ylim(0, 3)
ax.set_xticks([x + 0.5 for x in range(3)])
ax.set_yticks([2.5 - x for x in range(3)])
ax.set_xticklabels([LABELS[t] for t in TARGETS], fontsize=13)
ax.set_yticklabels([LABELS[s] for s in SOURCES], fontsize=13)
ax.set_xlabel("目标模型（后缀攻击谁）", fontsize=14)
ax.set_ylabel("来源模型（后缀从哪学来）", fontsize=14)
ax.tick_params(length=0)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
cb.set_label("越狱成功率 ASR（%）", fontsize=12)

plt.title("GCG 对抗后缀跨模型迁移攻击热力图\n（判定标准与原始实验一致：生成文本不含拒绝前缀）",
          fontsize=15)
plt.tight_layout()
fig1 = os.path.join(OUT, "fig3_transfer_heatmap.png")
plt.savefig(fig1, dpi=160)
plt.close()
print("已保存:", fig1)

print("\n[3x3 迁移矩阵]")
for s in SOURCES:
    row = []
    for t in TARGETS:
        v = MAT["%s->%s" % (s, t)]
        row.append("%d/%d" % (v["num"], v["denom"]))
    print("  %-9s %s" % (s, row))

