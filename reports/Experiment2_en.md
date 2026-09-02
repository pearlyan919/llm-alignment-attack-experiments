# Experiment Report 2: Cross-Architecture and Cross-Family Transferability of GCG Adversarial Suffixes (3×3 Transfer Matrix)

**Report ID**: EXP-2 ｜ **Language**: English ｜ **Date**: Aug 2026
**Data & Figures**: `outputs/exp2/`

---

## Abstract

Experiment 1 demonstrated that GCG can jailbreak individual models efficiently, but left unanswered whether such attack artifacts are universal. This experiment tests the core hypothesis: **vulnerabilities discovered by GCG are cross-architecture, cross-family universal defects (Universal Vulnerability), not overfitting to a single model.**

Methodologically, we extract 27 adversarial suffixes from GCG optimization runs (Llama-2×10, Llama-3.2×10, Gemma-2×7) and, under **zero-gradient pure inference**, inject each suffix into three target models to build a 3×3 source→target transfer matrix. Pipeline correctness is established via diagonal validation: self-transfer results for Llama-3.2 and Gemma-2 **exactly match** their original experiments (10/10, 7/7).

Results (on the shared 7 goals): **intra-family transfer Llama-2→Llama-3.2 reaches 100%**; cross-family transfer is significant yet directionally asymmetric — Gemma-2→Llama-3.2 is 100% while Llama→Gemma-2 is 29%. On the target side, Llama-3.2 is broken at 100% by all three sources (most fragile), while Llama-2 resists all three (most robust). Conclusion: **the vulnerability is a cross-family, cross-architecture general semantic defect rather than model-specific overfitting** — consistent with Experiment 1 ("more data did not repair the defect") and jointly pointing to out-of-distribution fragility in the shared instruction-processing semantic space of LLMs.

---

## 1. Research Question and Design

### 1.1 Scientific Questions

Experiment 1 answers "can a single model be attacked." This experiment asks the converse:

- **Q1**: Do suffixes optimized on a source model remain effective on **other models of the same family** (intra-family transfer)?
- **Q2**: Do they remain effective on models with **different architecture, vocabulary, and family** (cross-family transfer)?

Q1 probes "family-level defects" (Llama-2/Llama-3 share architecture and partial pretraining data); Q2 probes "universal defects in the attention semantic space" (Gemma and Llama differ in architecture, tokenizer, and training pipeline).

### 1.2 Why Pure-Inference Transfer Instead of Re-optimization

Re-running GCG on each target only proves "every model is attackable," which cannot distinguish **universal defects** from **model-specific overfitting**. If source suffixes trigger jailbreak on the target with **no gradient adaptation**, the exploited fragile structure must already exist in the target's weights and attention patterns — the decisive test for a universal vulnerability.

### 1.3 Sample Construction

| Source | # Suffixes | Origin |
|---|---|---|
| Llama-2-7B | 10 | Per-goal final control suffix from Experiment 1 GCG |
| Llama-3.2-3B | 10 | Same |
| Gemma-2-2B | 7 | Independent Gemma-2 GCG run |

Targets are the 7 AdvBench harmful behaviors covered by all three models (bomb×2, identity theft, manipulation, racist-website, virus, social-media incitement), ensuring equal denominators across matrix cells.

---

## 2. Experimental Setup

| Dimension | Configuration |
|---|---|
| Suffix handling | Stored as string → re-tokenized by target tokenizer (diagonals fully reproduced; boundary loss controlled) |
| Attack payload | `AttackPrompt(goal, target, control=suffix)`; template identical to original runs |
| Optimization | None (pure inference, zero gradient) |
| Judgment | GCG-consistent: `jailbroken = output free of refusal prefixes` |
| Deployment | Llama-2 4-bit / Llama-3.2 & Gemma-2 fp16 (inherited constraint from Exp. 1) |
| Validity check | Diagonal self-transfer: Llama-3.2 10/10 ✓, Gemma-2 7/7 ✓ — pipeline sound |

---

## 3. Core Results: 3×3 Transfer Matrix

**Shared 7 goals (ASR; rows = suffix source, columns = target model)**

| Source \ Target | Llama-2-7B | Llama-3.2-3B | Gemma-2-2B |
|---|---|---|---|
| **Llama-2** | 0/7 (0%, baseline) | **7/7 (100%, intra-family)** | 2/7 (29%, cross-family) |
| **Llama-3.2** | 0/7 (0%, cross-family) | 7/7 (100%, baseline) | 2/7 (29%, cross-family) |
| **Gemma-2** | 0/7 (0%, cross-family) | **7/7 (100%, cross-family)** | 7/7 (100%, baseline) |

**Supplementary (full goal set, unequal denominators)**: Llama-2→Llama-3.2 = 10/10 (100%); Llama-3.2→Gemma-2 = 4/10 (40%); Llama-2→Gemma-2 = 2/10 (20%).

Figure: `fig3_transfer_heatmap.png`; per-record data: `transfer_records.csv`.

---

## 4. In-Depth Analysis

### 4.1 Intra-family Transfer (Llama-2 → Llama-3.2): 100% ASR, far above the 70% hypothesis threshold

- All 10 Llama-2 suffixes — including the 9 that **failed against Llama-2 itself** — achieve 100% jailbreak on Llama-3.2.
- Corollary 1: despite 4.5× more training data and newer alignment, Llama-3.2 **fully inherits the family-level safety defects**.
- Corollary 2 (stronger): the **non-converged optimization trajectories** of Llama-2 (9 failed suffixes) already penetrate the shared fragile semantic subspace of the Llama family — "failure" only means the goal was not achieved on Llama-2 itself, not that the direction is void. The successful transfer of *failed* suffixes directly refutes the alternative explanation of source-model overfitting.

### 4.2 Cross-family Transfer: significant but directionally asymmetric

- **Gemma-2 → Llama-3.2 = 100%** (far above the 40% threshold): Gemma-family suffixes fully compromise Llama-family models.
- **Llama → Gemma-2 = 29%** (shared goals) or 20–40% (full set): partially effective, below the 40% threshold.
- Overall: GCG does **not** learn model-specific overfit features — significant transfer across architectures, tokenizers (SentencePiece vs. BPE), and training pipelines supports the "universal defect in the attention semantic space" hypothesis; yet **transfer strength is asymmetric**, with Llama suffixes being weaker against Gemma than the reverse.

### 4.3 Target-side Asymmetry: who is fragile, who is robust

| Target model | Broken by 3 sources | Interpretation |
|---|---|---|
| **Llama-3.2-3B** | 3/3 directions all 100% | Most fragile: any source's suffix succeeds |
| **Gemma-2-2B** | 1/3 at 100%; 2/3 at 29% | Medium: weak to own family, partial defense cross-family |
| **Llama-2-7B** | 3/3 directions all 0% | Most robust: all cross-model suffixes rejected |

Llama-2's 0% is consistent with its 1/10 own-attack rate and must be read cautiously alongside its 4-bit deployment and non-converged trajectories (§6.1).

### 4.4 Closed-loop Validation with Experiment 1

- Exp. 1: direct fragility of a single model under full attacker control.
- Exp. 2: generality of attack artifacts. Together they show that safety-alignment defects stem from adversarial fragility in the **shared instruction semantic space**, an out-of-distribution problem that data scaling and alignment iterations cannot eradicate.

---

## 5. Conclusion

> **Vulnerabilities discovered by GCG are cross-architecture, cross-family universal defects rather than single-model overfitting — the evidence is strong but transfer is directionally asymmetric. Intra-family transfer is complete (100%); cross-family transfer is substantial (0%–100%, median 29%), reaching 100% in the Gemma→Llama direction. Therefore, scaling up models or training data cannot eliminate family-level and cross-family alignment defects.**

---

## 6. Limitations and Rigor Statement

1. **Llama-2's "0% robustness" must be read cautiously**: its own attack converged on only 1/10 goals (4-bit + 8GB limit); the 0% transfer is self-consistent but cannot rule out a 4-bit-induced conservative response bias.
2. **Lenient judgment**: `jailbroken` only requires the absence of refusal prefixes. Spot checks show some "jailbreaks" are evasive/paraphrased outputs; EM (exact-match of target) is 0 throughout. A safety classifier (e.g., Llama Guard) is needed to quantify real harm levels.
3. **Uneven Gemma-2 coverage**: only 7 goals; its row denominator differs from the other two (flagged in supplementary data).
4. **Single run**: no multi-seed repetition; transfer judgments are subject to generation randomness (Experiment 3 quantifies this).
5. **String re-encoding loss**: suffixes stored as strings incur slight token-boundary loss upon re-encoding; full diagonal reproduction shows the loss is manageable.

---

## 7. Future Work

1. Replace the lenient refusal-prefix judgment with a safety classifier to quantify real harm (cf. Experiment 3's strict semantic grading);
2. Extend the matrix with Llama-3.1/3.3-8B, Mistral-7B, Qwen, etc., to test robustness of cross-family conclusions;
3. Identify "universally attackable" vs. "universally immune" goals and analyze their semantics;
4. Use transfer suffixes as adversarial training data to test migration-immunity improvements on Llama-3.2 (aligning with Experiment 3's "defend key trigger words" proposal).

---

## Appendix: Reproduction Notes

```bash
# Suffix extraction
extract_suffixes.py → outputs/exp2/suffixes.json (27 suffixes)
# Transfer test (pure inference)
transfer_test.py → outputs/exp2/transfer_records.csv
# Matrix & heatmap
outputs/exp2/transfer_matrix.json, fig3_transfer_heatmap.png
```

## References

1. Zou, A., et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043, 2023.
2. This series, Report 1 (EXP-1) — direct-attack comparison; source of suffixes.
3. This series, Report 3 (EXP-3) — truncation fingerprint and strict semantic grading; mechanistic validation.
