# TrustXAI-Derm: Validated Architecture & Model Selection (v1)

Literature-checked against live search (June 2026), not just the uploaded docs.

## 1. Novelty check against existing prior art

| Component | Confirmed prior art | Verdict |
|---|---|---|
| TIxAI (lesion-vs-background Grad-CAM relevance gap) | Ieracitano et al. 2025, *TIxAI: A Trustworthiness Index for eXplainable AI in skin lesions classification* (ScienceDirect/Neurocomputing), EfficientNet-B0, ISIC | **Not novel as-is.** Must be cited as the base metric, not reinvented as "ERI." |
| Generic "trust index from XAI fidelity+alignment+compliance" | TAXAI (Sci. Reports, Apr 2026) — radiology/pathology, general medical imaging, Trust Index 0.85–0.94 | **Domain-general framing is taken.** Survives only if specialized: dermoscopy + class-conditional + failure-prediction, which TAXAI does not do. |
| Conformal selective prediction + cost-aware deferral for clinical triage | Sci. Reports Feb 2026 — **sepsis/ICU triage**, not dermoscopy | **Open gap.** No dermoscopy-specific conformal deferral paper found. GAP8 survives. |
| Multi-XAI disagreement (Grad-CAM vs SHAP vs LIME) as a *reliability signal* | Literature (2024-25 reviews) repeatedly states methods disagree and *should* be combined, but no dermoscopy paper turns disagreement into a quantified per-sample index used for deferral | **Open gap as an idea — but the empirical test came back null (2026-07-05 run).** XDI (GAP-4) does not separate correct from incorrect predictions (Mann-Whitney p=0.98 on the full 990-image test set, holds across 3 binarization thresholds) and is outperformed by both softmax confidence and MC-Dropout entropy as a deferral signal. The novelty claim (no prior dermoscopy paper tested this) survives; the *finding* is negative. Demoted to a diagnostic/exploratory tool — see `GAP4_XDI_Disagreement_Report.md`. |
| Predicting explanation failure before computing it (meta-model on features → P(explanation unreliable)) | MetaQuantus (2023) addresses meta-evaluation of *XAI metrics*, not per-sample failure prediction in a clinical pipeline | **Open gap**, distinct enough from MetaQuantus. |
| DenseNet121 chosen for Grad-CAM spatial coherence vs ConvNeXt-Tiny/EfficientNetV2-S for raw accuracy | 2024-26 benchmarks: ConvNeXt-Tiny wins balanced accuracy (~77% BalAcc / F1 0.75 on ISIC-style multi-class); DenseNet121 wins Grad-CAM localization consistency, esp. on visually similar classes | Confirms documents — **trade-off is real, not fabricated.** |

**Conclusion:** the surviving, literature-defensible novelty is not "a new trust metric" (TIxAI/TAXAI already exist) — it is **the dermoscopy-specific, class-conditional, failure-predictive, multi-signal deferral pipeline that turns existing reliability metrics into an actionable clinical decision**, with explicit per-class and per-skin-tone breakdowns. That is what the rest of this document builds.

## 2. Model selection — DenseNet121, with ConvNeXt-Tiny as a reported accuracy baseline

**Primary backbone: DenseNet121** (ImageNet-pretrained, fine-tuned).

Why, with evidence:
- Confirmed in 2024-26 benchmarks to produce more spatially coherent, lesion-focused Grad-CAM/Grad-CAM++ maps than ConvNeXt/EfficientNetV2, *especially on visually similar minority classes* — this is the exact property the trust/explanation pipeline depends on (GAP1/GAP3/GAP4 need CAMs that meaningfully localize the lesion, not just classify it).
- Dense skip-connections give multiple feature-map resolutions for free, useful for Grad-CAM++ at different depths (needed for GAP5's saliency-geometry features).
- TIxAI's original paper used EfficientNet-B0 — using a different but comparably-sized backbone (DenseNet121) avoids a direct replication and lets us argue the metric generalizes across architectures, which is itself a minor but real contribution (cross-architecture validation of TIxAI was not done in the original paper).

**Reported secondary backbone: ConvNeXt-Tiny.**
Train it identically, report its higher raw balanced accuracy, and show — this is the actual paper hook — that **higher accuracy does not buy higher explanation trustworthiness**: ConvNeXt-Tiny's higher BalAcc should co-occur with no improvement (or a measurable drop) in TIxAI/XDI scores on minority classes. This accuracy-vs-trust dissociation is a clean, citable, falsifiable empirical claim and is novel as far as the search above could verify.

Do not use PanDerm, DINOv2, or EfficientNetV2-S as primary backbones — each appears in only one source document, none has comparative justification, and adding a foundation model multiplies engineering cost without literature support that it's better for *this* specific trust-prediction task. DINOv2 stays out of the core classification pipeline; note it is already used as a **retrieval encoder** (not a classifier) in GAP-10's executed contrastive analysis (off-the-shelf, not fine-tuned — see that report's Implementation Status table for what's built vs. proposed). GAP-7's own real, executed analysis does NOT use DINOv2 — it uses an ITA colorimetric proxy directly on HAM10000 (see below); GAP-7's report additionally *proposes* DINOv2/foundation-model backbones for a future gold-standard-labeled study, which remains unimplemented Future Work, distinct from what actually ran.

## 3. Unified pipeline

```
                         ┌─────────────────────────┐
                         │  HAM10000 + ISIC2018     │
                         │  masks (train/val/test)  │
                         └────────────┬─────────────┘
                                      │
                     preprocessing + augmentation
                     (hair removal, CLAHE, resize 224,
                      class-balanced sampling, no test leakage)
                                      │
                     ┌────────────────┴────────────────┐
                     │                                  │
            DenseNet121 (primary)              ConvNeXt-Tiny (baseline)
            Focal Loss, MC Dropout              same recipe, reported only
                     │                                  │
                     └────────────────┬─────────────────┘
                                      │
                         Prediction + softmax confidence
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
        Grad-CAM++                  SHAP                Per-class TIxAI
        (primary CAM)          (feature attribution)     (lesion-vs-bg
              │                       │                   relevance gap,
              └──────────┬────────────┘                   per GAP4)
                          │                                    │
                 Multi-XAI Disagreement Index (XDI)            │
                 XDI = 1 − IoU(Grad-CAM++ region, SHAP region)  │
                          │                                    │
                          └────────────────┬───────────────────┘
                                           │
                         Explanation Reliability Feature Vector
                         [TIxAI, XDI, MC-Dropout variance,
                          confusion-saliency distance (GAP3),
                          saliency-geometry descriptors (GAP5)]
                                           │
                         Failure-Prediction Meta-Model (GAP2,
                         gradient-boosted trees → P(explanation
                         is unreliable for this sample))
                                           │
                         Composite Clinical Risk Score (GAP8):
                         f(malignancy prior, prediction confidence,
                           conformal prediction set size,
                           explanation-reliability score)
                                           │
                              ┌────────────┴────────────┐
                              │                          │
                      Below deferral threshold   Above deferral threshold
                              │                          │
                     AI diagnosis + CAM shown      DEFER to clinician,
                     to clinician as support       flag WHY (low TIxAI /
                     (trusted explanation)          high XDI / OOD conformal
                                                     set / rare class)
```

## 4. What each stage answers scientifically

1. **Backbone comparison** → "Does higher accuracy imply higher explanation trust?" (Preliminary yes per the 2026-07-05 run, but that comparison is currently confounded by DenseNet121 being severely undertrained relative to ConvNeXt-Tiny — CELL 16/17b now gate this "headline" framing on both models reaching genuine convergence before it's reported as validated. See `GAP1_Model_Development_Report.md` §2.5.)
2. **TIxAI per-class (GAP1)** → "Which lesion classes get trustworthy explanations and which don't?" Real result (2026-07-05): classes differ significantly (Kruskal-Wallis p=2.95e-05), but NOT via a clean rarity gradient — DF (2nd-rarest) scores among the highest, not lowest, and class-frequency vs. TIxAI is not significantly correlated (Spearman ρ=0.11, p=0.82). See `GAP1_Model_Development_Report.md` §2.5 for the full reconciliation against this section's original "DF/VASC predicted worst" hypothesis.
3. **XDI (Grad-CAM++ vs SHAP disagreement, GAP4)** → "When two independent explanation methods disagree, is the prediction less trustworthy?" **Real result (2026-07-05): no.** Mann-Whitney p=0.98 (not significant, holds across 3 thresholds), and XDI underperforms both confidence and MC-Dropout entropy as a deferral signal. Novel as an untested idea; negative as an empirical finding. Demoted to diagnostic/exploratory use only — see `GAP4_XDI_Disagreement_Report.md`. Do not describe XDI as "the strongest surviving novel claim" in any downstream materials; that description predates execution and is no longer accurate.
4. **Confusion-Saliency Matrix (GAP3)** → diagnostic/qualitative companion, not a standalone claim — use as a figure, not a contribution. Note: the 2026-07-05 confusion counts (dominated by NV→MEL, 192 cases) reflect a severely undertrained DenseNet121 checkpoint and should be regenerated after the scheduler-bug fix + full retrain before being treated as a stable failure-mode audit.
5. **Failure-prediction meta-model (GAP2)** → "Can we flag an unreliable explanation *before* a clinician sees it, using only cheap signals (confidence, entropy, dropout variance) without computing the full TIxAI/SHAP pipeline at inference time?" High clinical value (real-time deployability). Real result: AUC-ROC=0.8324 (95% BCa CI [0.7933, 0.8657]), computed via a leakage-free per-fold Pipeline — this one holds up.
6. **Composite Clinical Risk Score + conformal deferral (GAP8)** → the actual clinical decision layer; this is what makes the system "trust-aware" rather than just "trust-measuring." Use **split conformal prediction** (per the sepsis-triage paper's validated methodology, adapted to dermoscopy) for the deferral set, since that exact technique is now externally validated as sound for cost-aware clinical deferral. **Correction (2026-07-05):** the deployed CCRS formula previously included a confusion-pair-risk term keyed on the sample's own ground-truth label (a real circularity bug, now fixed to use only the predicted class + an aggregate historical error rate). The pre-fix "100% error rate in deferred cases" figure was an artifact of that leak, not a validated deferral capability — see `GAP8_Clinical_Deferral_Report.md` §7.3.1 for the full reconciliation. Re-verify deferral numbers after re-running CELL 15 with the fix.

## 5. Recommended notebook scope (first pass)

Collapse the original 11-notebook plan into 6 Kaggle-runnable notebooks, each independently executable from saved checkpoints:

1. `01_Data_Prep_and_EDA.ipynb` — HAM10000 + ISIC2018 mask alignment, class balance, leakage checks, augmentation pipeline.
2. `02_Train_DenseNet121.ipynb` and a near-identical `02b_Train_ConvNeXtTiny.ipynb` — same training recipe, Focal Loss, MC Dropout heads, checkpointing.
3. `03_Explainability_Grad-CAM_SHAP.ipynb` — generate and cache Grad-CAM++/SHAP maps for both models on the test set.
4. `04_TIxAI_and_XDI.ipynb` — compute per-class TIxAI (GAP1) and XDI (multi-XAI disagreement, GAP4) for both backbones; this produces the accuracy-vs-trust dissociation figure.
5. `05_Failure_Prediction_and_Deferral.ipynb` — train the GAP2 meta-model, build the conformal prediction sets, compute the GAP8 Composite Clinical Risk Score, run the deferral simulation.
6. `06_Statistical_Validation_and_Figures.ipynb` — bootstrap CIs, multiple-comparison correction, decision curve analysis, all publication figures.

GAP3 (confusion-saliency) folds into notebook 4 as a secondary figure rather than its own notebook. GAP5 (saliency-geometry) folds into notebook 4 as additional features feeding notebook 5's meta-model (real result, 2026-07-05: null — none of the 4 implemented descriptors separate correct from incorrect predictions, all p>0.32; see `GAP5_Saliency_Geometry_Report.md` §1.3).

**Status update (2026-07-05) — this "Future Work" framing is now partially stale.** GAP6, GAP7, and GAP10 have each since had a real (if scoped-down or proxy-based) analysis executed and reported, not just proposed:
- **GAP6** (retitled from "Longitudinal Consistency" to "Temporal/Stochastic Saliency Stability" — the notebook never had a multi-visit longitudinal dataset available) ran: median STS=0.850, 7% of images show unstable (non-reproducible) explanations under repeated stochastic inference.
- **GAP7** ran a directional ITA-proxy skin-tone audit on HAM10000 itself (not a separate future paper): no significant TIxAI gap (p=0.20) but a significant uncertainty gap (p=0.0004) across estimated FST buckets.
- **GAP10** ran its core DINOv2-retrieval contrastive analysis on HAM10000's own test set: 1,528 pairs categorized, but the central hypothesis (embedding similarity predicts saliency similarity, target ρ>0.60) came back null (ρ=0.024, p=0.73).
- **GAP9 alone remains genuinely not-run** — its diffusion pipeline failed to import (transformers/diffusers version skew); a fix has been applied to the pin but not yet confirmed on a live session.

Each of GAP6/7/9/10 also still has a larger, un-executed "proper" version proposed in its own report (external gold-standard datasets, fine-tuned encoders, spatial registration, clinical validation) — those larger versions remain genuine Future Work / candidate separate papers. The point of this update is only that "Future Work" no longer means "nothing has been tried" for three of the four.

## Sources consulted

- Ieracitano et al., "TIxAI: A Trustworthiness Index for eXplainable AI in skin lesions classification," ScienceDirect/Neurocomputing 2025 — https://www.sciencedirect.com/science/article/pii/S092523122500373X
- "The Trust-Aware XAI (TAXAI) framework," Scientific Reports, Apr 2026 — https://www.nature.com/articles/s41598-026-44167-3
- "Conformal selective prediction with cost aware deferral for safe clinical triage under distribution shift," Scientific Reports, Feb 2026 — https://www.nature.com/articles/s41598-026-40637-w
- "Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics," arXiv 2503.23819
- Benchmarking study: ConvNeXt-Tiny vs EfficientNetV2-S vs DenseNet121 on imbalanced skin lesion classification with Focal Loss + Grad-CAM (2024-26), BalAcc/F1/Grad-CAM localization comparison
- "Explainable Artificial Intelligence for Skin Lesion Classification: A Comprehensive Review of Methods and Challenges," MDPI Technologies 14(7):391
