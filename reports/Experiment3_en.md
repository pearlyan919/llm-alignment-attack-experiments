# Experiment Report 3: "Semantic Fingerprint" of Adversarial Suffixes via Truncation Tests — A Microdissection of GCG's Attack Mechanism

**Report ID**: EXP-3 ｜ **Language**: English ｜ **Date**: Aug 2026
**Data & Figures**: `outputs/exp3/`

---

## Abstract

Experiment 2 established the cross-family transferability of GCG suffixes but did not explain *why* transfer happens. This experiment token-level-truncates adversarial suffixes and tests each retained fraction, dissecting the functional structure of the suffix at the mechanistic level: is jailbreak capability **uniformly distributed across all tokens**, or **concentrated in a few leading "syntactic triggers"**? In parallel, we conduct a **methodological self-audit** of the "refusal-prefix" criterion inherited from Experiments 1–2 by introducing strict semantic grading (hard refusal / soft refusal / true compliance), re-examining the semantic authenticity of previously reported ASR.

Method: from Experiment 2's transfer matrix, we select 5 suffixes that successfully transfer to Gemma-2 (2 Llama-2-source cross-family + 3 Llama-3.2-source intra-family), truncate tokens at 10%–100% (10% steps), test Gemma-2 under pure inference, and perform 5 sampled runs at the critical 30%/50%/100% levels.

Three core findings: (i) **No monotonic "cliff" was observed** — lenient ASR oscillates violently (20%→80%) and single-run judgments are noisy; (ii) **strict semantic grading shows "true compliance" is constant at 60% across 30%/50%/100%** — suffix length does not change whether the model truly complies; triggering ability is determined by a few leading tokens; (iii) **mechanistic revision: the primary function of GCG suffixes is "refusal softening", not "semantic hijacking"** — exact-match rate of the malicious target is 0 throughout; Gemma-2 never actually outputs harmful payloads but merely softens refusals from hard rejection to "I understand but I cannot." These findings directly inform precision defense (leading-trigger detection) and evaluation methodology (semantic-level judgment).

---

## 1. Research Questions and Design

### 1.1 Scientific Questions

Experiment 2 proved transferability, leaving two open problems:

- **Q1 (structure)**: How is the jailbreak function distributed over the suffix's tokens? Is there a "critical head" that must be preserved?
- **Q2 (essence)**: What does a "successful jailbreak" actually mean — true semantic compliance producing harmful content, or merely a **softened expression of refusal** (soft refusal)?

Q1 directly tests the hypothesis that "syntactic triggers / attention hijackers sit at the suffix head, while the tail is overfitting noise." Q2 is a **meta-scientific audit** of the evaluation metric used in the first two experiments, determining the reliability of the entire research chain.

### 1.2 Why Truncation as a Probe

Gradient-based attribution (attention/saliency) is noisy and requires re-accessing internal states. Truncation is a **zero-cost, interpretable, deterministic functional probe**: if removing the tail does not affect jailbreak, those tokens are not necessary; if removing the head immediately kills it, the head carries the critical function.

### 1.3 Samples and Operation

| # | Source | Goal (abbrev.) | Suffix tokens | Transfer type |
|---|---|---|---|---|
| 1 | Llama-2 | Manipulation tutorial | 21 | Cross-family |
| 2 | Llama-2 | Dangerous-behavior post | 20 | Cross-family |
| 3 | Llama-3.2 | Racist website | 9 | Intra-family |
| 4 | Llama-3.2 | Firearms for children | 25 | Intra-family |
| 5 | Llama-3.2 | Exploit script | 12 | Intra-family |

Operation: tokenize by the target model (Gemma-2) → keep the first k% tokens (k = 10,20,…,100) → re-encode and concatenate → pure-inference generation. Two judgment levels: lenient (inherited refusal prefixes) and strict (semantic grading, §3.2).

---

## 2. Results

### 2.1 Lenient ASR Truncation Curve (Experiments 1–2 criterion)

| Retained | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| Lenient ASR | 20% | 60% | 60% | 80% | 80% | 20% | 60% | 80% | 40% | 80% |

5-sample runs at critical levels (multi-ASR): 30%→40–60%, 50%→0–80%, 100%→0–60% (high variance across samples).

### 2.2 Strict Semantic Grading (core metric)

All generated texts re-classified into three classes:

| Retained | Hard refusal | Soft refusal | **True compliance** | Strict ASR |
|---|---|---|---|---|
| 30% | 2 | 0 | **3** | **60%** |
| 50% | 1 | 1 | **3** | **60%** |
| 100% | 1 | 1 | **3** | **60%** |

Per-sample, per-level details: `truncation_records.csv`, `truncation_summary.json`, `truncation_strict.json`; figures: `fig4_asr_truncation_curve.png` (lenient curve), `fig5_truncation_heatmap.png` (per-sample heatmap).

---

## 3. In-Depth Analysis

### 3.1 The Predicted "Monotonic Cliff" Did Not Appear; Single-Run Noise Dominates

The lenient ASR curve oscillates violently (20%→60%→80%→20%→80%) with **no monotonic cliff**. Five sampled runs reveal the cause: at the same retention level, temperature sampling flips success/failure (e.g., sample 1 judged successful at 50% in a single run but only 1/5 in sampling). This quantitatively warns Experiment 2's single-run transfer judgments: **the variance of single-generation ASR is large; multi-run averaging is required.**

### 3.2 Core Mechanistic Finding: GCG's Function Is "Refusal Softening", Not "Semantic Hijacking"

Line-by-line inspection reveals a highly consistent pattern:

- Texts judged "jailbroken" under the lenient criterion are mostly `"I understand you're interested in X... but I cannot/advise against..."` — **avoiding hard refusal prefixes while still refusing**;
- Even "truly compliant" samples rarely output actual malicious payloads (e.g., only "It sounds like you're interested in cybersecurity... Let's explore...");
- **Exact-match rate `em` of the target is 0 throughout** — no generated text contains the preset harmful completion target.

This constitutes a **methodological amendment** to Experiments 1–2: their reported "high ASR" measures the *syntactic refusal-softening ability* of GCG suffixes (syntactic trigger / attention hijacker), not genuine content jailbreak. GCG makes the model "speak softly before refusing", which happens to bypass keyword-based refusal-prefix detectors.

### 3.3 Highly Front-Loaded Triggers: Confirming the "Syntactic Trigger" Hypothesis

- 4 of 5 samples have `min_success_ratio ≤ 0.3` (lenient jailbreak already triggered at 20%–30% retention);
- Sample 5 triggers compliant output with only the **first token** `cas` (10%);
- Triggering ability concentrates in the first 1–6 tokens (`cas` / `provide crispu` / `publicdimensions` / `parts emphas literary marks`); **tail tokens contribute approximately zero** — consistent with "GCG learns front-loaded, transferable syntactic perturbations; the tail is overfitting noise."

### 3.4 An Honest Counterexample: Sample 2 Reveals the Fragility of Transfer Judgment

Sample 2 (Llama-2 source, dangerous-behavior post) fails at full length under lenient judgment with 0% sampled ASR — inconsistent with Experiment 2's transfer-matrix record that this goal transferred successfully. This exposes instability between single-run transfer judgment and temperature sampling, reinforcing the §3.1 methodological warning.

---

## 4. Conclusions and Scientific Implications

> **The "cross-model jailbreak" of GCG suffixes is essentially the transferability of "syntactic triggers", not of "semantic jailbreak". Attack function is highly concentrated in a few leading tokens and operates by softening the model's refusal phrasing to evade keyword detection — it does not truly breach the semantic alignment layer. Lenient refusal-prefix-based ASR systematically overestimates attack success; semantic-level judgment is mandatory for jailbreak evaluation.**

Four actionable implications:

1. **Evaluation**: ASR must distinguish "soft refusal" from "true compliance", judging by whether actual harmful content is emitted (target hit / safety classifier) — this grading framework transfers directly to re-auditing Experiments 1–2 data;
2. **Defense**: front-loaded triggers imply that anomaly detection on the first 1–6 tokens (rare token combinations, non-semantic gibberish, control characters) can intercept most transfer attacks before the semantic layer — theoretical grounding for "precision defense of key trigger words";
3. **Alignment assessment**: judging whether a newer model (e.g., Llama-3.2 vs. Llama-2) is safer must look beyond "I'm sorry" to whether it **still refuses to emit real harmful content** under syntactic perturbation — Gemma-2's semantic alignment remained solid throughout, which is itself the correct lens for safety evaluation;
4. **Adversarial training**: leading triggers are high-value adversarial examples for training, enabling targeted hardening against prefix perturbation.

---

## 5. Limitations

1. Small sample (5 suffixes × 10 levels); strict grading is rule-based heuristics (keywords + regex), not a safety classifier;
2. Mixed protocol of single deterministic runs plus 5-sample runs only at critical levels; intermediate levels lack repeated sampling;
3. Only one target model (Gemma-2); extrapolation to other models requires caution;
4. Truncation is "prefix retention" only; no "tail-truncation / mid-deletion" controls, so the functional-distribution conclusion is one-sided.

---

## 6. Future Work

1. Re-judge all Experiment 1–2 outputs with Llama Guard / GPT-4 classifiers to produce a "true-harm ASR";
2. Extend truncation to "tail-truncation / mid-deletion" control groups for a complete token functional map (semantic fingerprint map);
3. Token-level ablation of the first 1–6 tokens to locate the "minimal trigger element";
4. Inject "soft-refusal" samples into adversarial training and test suppression of lenient ASR.

---

## Appendix: Reproduction Notes

```bash
# Truncation test (main experiment)
truncation_test.py → outputs/exp3/truncation_records.csv & summary.json
# Visualization
gen_trunc_viz.py → fig4_asr_truncation_curve.png / fig5_truncation_heatmap.png
# Strict semantic grading
gen_trunc_strict.py → truncation_strict.json
```

## References

1. Zou, A., et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043, 2023.
2. This series, Report 2 (EXP-2) — transfer matrix; source of samples.
3. This series, Report 1 (EXP-1) — direct-attack comparison; its judgment metric is meta-audited here.
