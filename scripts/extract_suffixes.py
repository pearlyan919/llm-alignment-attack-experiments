"""
实验二（步骤 1）：提取三个模型的最终攻击后缀 → outputs/exp2/suffixes.json
输入：results/ 下三个权威 GCG 结果 JSON（llama-2 / llama-3.2 / gemma-2）
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.OUT_EXP2
os.makedirs(OUT, exist_ok=True)

FILES = {
    "llama-2": paths.result_json("llama-2"),
    "llama-3.2": paths.result_json("llama-3.2"),
    "gemma-2": paths.result_json("gemma-2"),
}

all_suffixes = []
for tag, fn in FILES.items():
    d = json.load(open(fn, encoding="utf-8"))
    goals = d["params"]["goals"]
    targets = d["params"]["targets"]
    ctrls = d["controls"]
    tests = d["tests"]
    CK = len(tests) // len(goals)
    print("== %s: n_goals=%d CK=%d" % (tag, len(goals), CK))
    for i, g in enumerate(goals):
        # 最终 checkpoint 的后缀
        suffix = ctrls[i * CK + CK - 1]
        # 最终测试结果
        info = tests[i * CK + CK - 1].get(g)
        passed = int(info[0][1]) if info and info[0] else 0
        loss = info[0][3] if info and info[0] else None
        rec = {
            "source": tag,
            "goal": g,
            "target": targets[i],
            "suffix": suffix,
            "source_pass": passed,
            "source_loss": loss,
        }
        all_suffixes.append(rec)
        print("  G%-2d pass=%d loss=%.3f suffix=%r" % (
            i + 1, passed, loss if loss is not None else -1, str(suffix)[:50]))

with open(os.path.join(OUT, "suffixes.json"), "w", encoding="utf-8") as f:
    json.dump(all_suffixes, f, ensure_ascii=False, indent=2)
print("\n已保存 %d 条后缀 → %s" % (len(all_suffixes), os.path.join(OUT, "suffixes.json")))

# 目标对齐检查（按 goal 全文匹配）
print("\n== 目标对齐检查 ==")
l3 = {r["goal"]: r for r in all_suffixes if r["source"] == "llama-3.2"}
g2 = {r["goal"]: r for r in all_suffixes if r["source"] == "gemma-2"}
common = set(l3) & set(g2)
print("共同目标数: %d" % len(common))
for goal in sorted(common):
    print("  [共有] %s" % goal[:55])

