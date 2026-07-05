# GAP #4 — MULTI-XAI DISAGREEMENT INDEX (XDI)

## **When Should Clinicians Not Trust an AI Explanation?**
### A Trust-Aware Framework for Dermoscopic Skin Cancer Diagnosis
### **Contribution: Grad-CAM++ vs. SHAP Disagreement as a Reliability Signal**

---

> **Document Type:** Results Report (post-execution)
> **Date:** 2026-07-05
> **Status:** Supersedes the earlier duplicate content that was mistakenly filed under this GAP number (`GAP4_PerClass_TIxAI_Report.md`, which actually described **GAP-1**'s per-class TIxAI topic and has been merged into `GAP1_Model_Development_Report.md`). Per the notebook's own cell titles, **GAP-1 = Per-Class TIxAI** (CELL 10) and **GAP-4 = Multi-XAI Disagreement Index** (CELL 12) — this file is now the correct home for GAP-4.

---

## 1. The Scientific Question

> **When two independent explanation methods (Grad-CAM++ and SHAP) disagree about which pixels drove a prediction, is that disagreement itself a signal that the prediction is unreliable?**

The Multi-XAI Disagreement Index (XDI) formalizes this as:

$$\text{XDI} = 1 - \text{IoU}(\text{CAM}_{\text{bin}}, \text{SHAP}_{\text{bin}})$$

where both maps are binarized at a top-k% activation-mass threshold before computing Intersection-over-Union. XDI ranges [0, 1]; higher XDI means the two explanation methods point at more different regions.

The motivating hypothesis: if a model's classification is unreliable, its two independent explanation mechanisms should also disagree about *why* — making XDI a cheap, model-agnostic proxy for "should a clinician double-check this case," without needing ground-truth lesion masks.

## 2. Real Result (test set, n=990, full coverage)

**Mean XDI: 0.591 ± 0.042.** Correct-prediction mean XDI = 0.594; incorrect-prediction mean XDI = 0.589 — nearly identical.

**Primary test — does XDI separate correct from incorrect predictions?**

Mann-Whitney U (correct XDI < incorrect XDI, one-sided): **U = 131,034, p = 0.9768.**

**This is a null result.** There is no evidence that XDI is lower for correct predictions than incorrect ones in this dataset — the two distributions are statistically indistinguishable in the hypothesized direction.

**Threshold-robustness check** (does a different binarization threshold change the conclusion?):

| Threshold (k) | Mann-Whitney p | Rank-biserial r | Significant (p<0.01)? |
|---|---|---|---|
| 25% | 0.545 | −0.004 | No |
| 50% | 0.977 | −0.073 | No |
| 75% | 0.662 | −0.015 | No |

The null result is not an artifact of the 50% threshold choice — it holds (fails to reach significance) at all three thresholds tested, with effect sizes indistinguishable from zero.

**Head-to-head against cheaper baselines** (area under the selective-accuracy curve — higher is a better deferral signal):

| Signal | AUSC |
|---|---|
| XDI | 0.394 |
| MC-Dropout entropy | 0.424 |
| Softmax confidence | 0.497 |

XDI is not just non-significant on the primary test — it is **outperformed by both cheaper, already-available baselines** (confidence and MC-Dropout entropy) at the task it was proposed for.

## 3. Conclusion and Status

**XDI does not beat baseline as a deferral/reliability signal.** Per this project's own pre-registered contingency plan for a null XDI finding: **XDI is demoted from "deferral signal" to a diagnostic/exploratory visualization only.**

Concretely, this means:
- XDI **should not** be used as an input to any clinical deferral decision (it is intentionally excluded from the Composite Clinical Risk Score's inputs for this reason — see `GAP8_Clinical_Deferral_Report.md`).
- XDI **may still be useful** as a qualitative, case-by-case visualization tool for developers/researchers auditing *why* a single prediction's two explanation methods disagree, independent of whether that disagreement predicts correctness in aggregate.
- No document in this project should describe XDI as a "validated finding," a "working reliability signal," or list it as a contribution on equal footing with GAP-1/GAP-2/GAP-8, which do show real, actionable signal. It is an honestly negative result, not a hidden bug — the code and statistics behind it were verified correct; the underlying hypothesis (Grad-CAM++/SHAP disagreement predicts correctness) simply was not supported by this model and dataset.

## 4. Caveats

- This result reflects the DenseNet121 checkpoint trained to 60/150 epochs under a scheduler bug that has since been fixed (see `GAP1_Model_Development_Report.md` §2.5). It is possible, though not guaranteed, that XDI behaves differently once the model is retrained to full convergence — re-run this test after that retrain before treating the null result as final for the corrected model.
- SHAP values here use `GradientExplainer` (not `DeepExplainer`), chosen for speed/memory reasons (see CELL 12's own inline documentation). This is a reasonable, standard SHAP approximation for CNNs, but a different SHAP estimator could in principle shift the exact XDI values, if not the qualitative conclusion.
- "XAI disagreement should track reliability" is an intuitive hypothesis, not a settled finding elsewhere in the literature for dermoscopy specifically — a null result here is a genuine, reportable contribution (most XAI papers only show cases where methods agree or disagree qualitatively, without testing whether disagreement is diagnostic at the population level).
