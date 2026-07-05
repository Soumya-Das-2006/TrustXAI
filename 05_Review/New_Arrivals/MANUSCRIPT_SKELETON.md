# TrustXAI-Derm — Manuscript Skeleton (fill from METRICS_TABLE.csv)

> RULES FOR FILLING THIS IN
> 1. Every `{{...}}` is a placeholder. Replace it ONLY with a number that appears
>    in a real cell output / METRICS_TABLE.csv from your Kaggle run.
> 2. If a placeholder's source metric shows "NOT COMPUTED", do not invent it —
>    either fix the cell and rerun, or cut the sentence.
> 3. Do not re-add GAP-4, GAP-5, GAP-9, or GAP-10 as positive contributions.
>    They are null / never-run; they belong only in Limitations / Future Work.
> 4. If Gate A did NOT reach ≥0.75 accuracy, say so plainly in Limitations —
>    do not round up or omit it.

---

## Title
When Should Clinicians *Not* Trust an AI Explanation? A Trust-Aware, Explanation-Conditioned Deferral Framework for Dermoscopic Skin Cancer Diagnosis

## Abstract
- One-line problem: high classification accuracy does not guarantee trustworthy explanations.
- What we built: DenseNet121 primary + ConvNeXt-Tiny baseline on HAM10000, with an
  explanation-reliability meta-model (GAP-2) and a non-circular composite clinical risk
  score + conformal deferral layer (GAP-8).
- Headline numbers (fill): meta-model AUC = {{gap2_auc}}; deferral covers {{gap8_defer_pct}}%
  of cases with a deferred-set error rate of {{gap8_deferred_err}}.
- Honest scope: DenseNet121 test accuracy = {{dense_acc}} on imbalanced HAM10000; this is a
  methods/framework contribution, not a new SOTA classifier.

## 1. Introduction
- Motivation: clinical decision support needs a "should I trust this?" signal, not just a label.
- Prior work covers TIxAI-style trust indices and conformal triage separately; none combine a
  dermoscopy-specific, class-conditional, *explanation-failure-predictive* deferral pipeline.
- Contributions (state only the ones that pass their gate):
  1. GAP-2: predict explanation unreliability from cheap internal features before computing SHAP.
  2. GAP-8: a non-circular composite risk score feeding conformal, cost-aware deferral.
  3. GAP-1 (supporting): per-class explanation reliability varies significantly across lesion classes.

## 2. Methods
- Data: HAM10000, 10,015 images, 7 classes; lesion-level (not image-level) train/val/test split
  ({{n_train}}/{{n_val}}/{{n_test}}) to prevent leakage.
- Models: DenseNet121 (primary, chosen for Grad-CAM++ spatial coherence) + ConvNeXt-Tiny (baseline).
- Training: AdamW + OneCycleLR. **Note a corrected implementation detail:** the LR scheduler is
  stepped once per batch (per OneCycleLR's contract); an earlier revision stepped it once per epoch,
  which prevented the schedule from annealing. All results here use the per-batch version.
- Calibration: post-hoc temperature scaling on a held-out calibration split (separate from test).
- Explanation signals: Grad-CAM++ (localization / TIxAI), SHAP (GradientExplainer), MC-Dropout
  (predictive-uncertainty).
- **Deferral risk score (GAP-8):** CCRS is a weighted combination of (1−confidence), (1−TIxAI),
  XDI, malignancy prior × (1−confidence), conformal-set-size, and a confusion-pair risk term.
  The confusion-pair term is keyed on the *predicted* class and an aggregate population-level
  error rate only — it does not use the ground-truth label of the case being scored. (An earlier
  revision did, which produced a degenerate deferral result; that is corrected here.)

## 3. Results
### 3.1 Classification & calibration
- DenseNet121: acc={{dense_acc}}, balanced acc={{dense_bacc}}, macro-F1={{dense_f1}},
  macro-AUC={{dense_auc}}, ECE={{dense_ece}}.
- ConvNeXt-Tiny: acc={{conv_acc}}, macro-AUC={{conv_auc}}.

### 3.2 Contribution 1 — Explanation-failure prediction (GAP-2)
- Meta-model 5-fold CV AUC-ROC = {{gap2_auc}} (95% BCa CI [{{gap2_ci_lo}}, {{gap2_ci_hi}}]).
- Interpretation: reliability can be flagged from cheap internal features, before the full
  Grad-CAM++/SHAP pipeline is computed.

### 3.3 Contribution 2 — Non-circular CCRS + conformal deferral (GAP-8)
- Deferral rate = {{gap8_defer_pct}}%; deferred-set error rate = {{gap8_deferred_err}};
  AI-handled (non-deferred) accuracy = {{gap8_ai_acc}}.
- Per-FST conformal coverage (target 90%): {{fst_coverage_summary}} (ITA-estimated buckets; caveat below).

### 3.4 Supporting — Per-class explanation reliability (GAP-1)
- Kruskal-Wallis across classes: p = {{gap1_kw_p}} (significant if <0.01).
- Note the pattern is class-dependent but NOT a clean rarity gradient: class-frequency vs.
  median-TIxAI Spearman ρ = {{gap1_freq_rho}} (p = {{gap1_freq_p}}, not significant).

## 4. Limitations
- **Classifier accuracy:** DenseNet121 reaches {{dense_acc}} on HAM10000; severe class imbalance
  limits rare-class recall. This is a framework paper, not a SOTA-accuracy claim.
- **Skin-tone labels are a proxy:** FST buckets are ITA-estimated from peri-lesional pixels, not
  clinically-assigned Fitzpatrick labels. GAP-7 found no significant TIxAI gap (p={{gap7_tixai_p}})
  but a significant predictive-uncertainty gap (p={{gap7_entropy_p}}) across estimated buckets;
  both need validation on real-FST datasets (PAD-UFES-20, Fitzpatrick17k, MSKCC).
- **Small-n exploratory metrics:** GAP-6 STS on n={{gap6_n}} images (median STS={{gap6_median}}).
- **Negative / unexecuted results reported honestly, not as contributions:** multi-XAI disagreement
  (XDI) did not separate correct from incorrect predictions (p={{gap4_p}}) and is used only as a
  diagnostic visualization; saliency-geometry descriptors showed no correct/incorrect separation
  (all p>0.32); the contrastive embedding↔saliency correlation was null (ρ={{gap10_rho}}); the
  synthetic dark-skin augmentation pipeline did not execute (dependency failure) and remains future work.
- **External validation pending:** PAD-UFES-20 not yet attached.

## 5. Conclusion
- A dermoscopy deferral pipeline whose deferral decision is explanation-aware and non-circular,
  with an inference-cheap reliability predictor. Reproducible; external validation is the next step.

## Appendix
- Table A1: full per-class metrics (DenseNet121).
- Table A2: per-FST conformal coverage.
- Figure A1: GAP-2 meta-model feature importances.
