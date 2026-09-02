"""
实验二（步骤 3）：基于共同目标重新统计 3x3 迁移矩阵（分母一致，可比性更强）
共同目标 = llama-2 与 gemma-2 都覆盖的 goal 全文（跨异构的公共基准）。
输入：outputs/exp2/suffixes.json + transfer_records.csv（transfer_test.py 产出）
输出：控制台打印（供 transfer_report 引用；不含独立产物文件）
"""
import json, os, csv
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.OUT_EXP2
SUF = json.load(open(os.path.join(OUT, "suffixes.json"), encoding="utf-8"))
rows = list(csv.DictReader(open(os.path.join(OUT, "transfer_records.csv"), encoding="utf-8-sig")))

# 共同目标 = llama-2 与 gemma-2 都覆盖的 goal 全文
l2 = {r["goal"] for r in SUF if r["source"] == "llama-2"}
g2 = {r["goal"] for r in SUF if r["source"] == "gemma-2"}
common = l2 & g2
print("共同目标 %d 条:" % len(common))
for g in sorted(common):
    print("  - %s" % g[:60])

SOURCES = ["llama-2", "llama-3.2", "gemma-2"]
TARGETS = ["llama-2", "llama-3.2", "gemma-2"]
mat = defaultdict(lambda: {"n": 0, "d": 0})
for r in rows:
    if r["goal"] not in common:
        continue
    k = (r["source"], r["target"])
    mat[k]["d"] += 1
    if r["jailbroken"] == "True":
        mat[k]["n"] += 1

print("\n[共同 %d 条目标上的迁移矩阵]" % len(common))
print("%-11s | %-12s | %-13s | %-11s" % ("Source\\Tgt", "Llama-2-7B", "Llama-3.2-3B", "Gemma-2-2B"))
for s in SOURCES:
    row = []
    for t in TARGETS:
        v = mat[(s, t)]
        row.append("%d/%d (%.0f%%)" % (v["n"], v["d"], 100.0 * v["n"] / v["d"] if v["d"] else 0))
    print("%-11s | %-12s | %-13s | %-11s" % (s, row[0], row[1], row[2]))

