# TrustXAI-Derm / FairSkinNet — Full Research Report
**Explainable Deep Learning-Based Skin Cancer Detection Using Dermoscopic Images**

> Researcher: Bhavesh Vaghela | PhD Program, Parul University, Vadodara
> Guide: Dr. (Prof.) Priya Swaminarayan
> Report compiled: July 2026 | Based on complete document analysis across all uploaded files

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What We Built — Architecture & Pipeline](#2-what-we-built)
3. [Datasets Used](#3-datasets-used)
4. [Saturated Components — What Is NOT Novel](#4-saturated-components)
5. [Forensic Novelty Audit — 8 Stages](#5-forensic-novelty-audit)
6. [Four Surviving Novel Contributions](#6-four-surviving-novel-contributions)
7. [Existing Literature — Paper-wise Analysis](#7-existing-literature)
8. [Research Gap Analysis](#8-research-gap-analysis)
9. [Comparison Table](#9-comparison-table)
10. [Common Problems Across Literature](#10-common-problems)
11. [Notebook Code Review — What Needs Fixing](#11-notebook-code-review)
12. [Validation Experiments & Statistical Tests](#12-validation-experiments)
13. [Novel Research Opportunities](#13-novel-research-opportunities)
14. [Research Gap Matrix](#14-research-gap-matrix)
15. [Novelty Score Summary](#15-novelty-score-summary)
16. [Reviewer Perspective (Reviewer #2)](#16-reviewer-perspective)
17. [Publication Strategy](#17-publication-strategy)
18. [Final Research Roadmap](#18-final-research-roadmap)
19. [Master Summary Table](#19-master-summary-table)

---

## 1. Project Overview

### Research Problem
Skin cancer detection via dermoscopic imaging using deep learning achieves high aggregate accuracy but fails in three co-occurring ways:

1. **Accuracy-explainability dissociation** — a model can classify correctly while producing heatmaps that attend to hair, rulers, or background instead of the lesion
2. **Class-level explanation inequality** — rare diagnostic classes (DF, VASC) are systematically worse-explained even when correctly predicted, but global metrics hide this
3. **No pre-generation warning** — there is no mechanism to predict whether a specific image's heatmap will be unreliable *before* running the expensive XAI pipeline

### Clinical Importance
- Skin cancer affects 3+ million patients/year globally
- Dermatoscopy AI is being deployed in clinical settings where clinicians trust heatmaps as decision aids
- A model scoring 0.71 globally on TIxAI may silently produce 0.34-quality heatmaps for DF — misleading clinicians without warning
- No currently deployed system measures or reports explanation reliability per diagnostic class

### Research Hypothesis
> Accuracy and explanation trustworthiness are dissociable, that dissociation is class-dependent and measurable, and it is predictable in advance using features already available at inference time.

---

## 2. What We Built

### Architecture: FairSkinNet

```
Input Image (224×224)
        ↓
EfficientNetV2-S Backbone (ImageNet pretrained)
        ↓ (1280-dim feature vector)
  ┌─────────────────────────────────┐
  │     Cross-Attention Fusion       │
  │  Query/Value = image features    │
  │  Key         = metadata features │
  └─────────────────────────────────┘
        ↓ (256-dim fused representation)
MC Dropout (active at inference for uncertainty)
        ↓
Classifier (→ 7 classes)
        ↓
┌──────────────────────────────────────────┐
│           Output Layer                    │
│  • Prediction + softmax confidence        │
│  • Grad-CAM++ heatmap                    │
│  • SHAP attribution                      │
│  • TIxAI per class (novel)               │
│  • XDI disagreement score (novel)        │
│  • Failure prediction flag (novel)       │
│  • Confusion Saliency Matrix (novel)     │
└──────────────────────────────────────────┘
```

### Metadata Stream (Parallel Branch)
- Input: age (normalised), sex (binary), anatomical site (one-hot), Fitzpatrick type (ordinal), lesion diameter
- Architecture: Linear(n_meta→64) → ReLU → Dropout(0.3) → Linear(64→128) → ReLU
- Fused via MultiheadAttention (4 heads, 256-dim) with residual + LayerNorm

### Training Recipe
| Component | Choice | Source |
|---|---|---|
| Backbone | EfficientNetV2-S | Tan & Le, ICML 2021 |
| Loss | Focal Loss (α=0.25, γ=2.0) | Lin et al., ICCV 2017 |
| Optimiser | AdamW (lr=3e-4, wd=1e-4) | Loshchilov & Hutter, ICLR 2019 |
| LR Schedule | Cosine annealing with warm restart | Loshchilov & Hutter, ICLR 2017 |
| Augmentation | RandomFlip, RandomRotation±20°, ColorJitter, RandomErasing | Standard |
| Early stopping | patience=10, monitor val macro-F1 | — |

---

## 3. Datasets Used

| Dataset | Images | Classes | Key Feature | Role |
|---|---|---|---|---|
| HAM10000 | 10,015 | 7 (MEL, NV, BCC, AKIEC, BKL, DF, VASC) | Dermoscopy; has age/sex/site metadata | Primary training |
| ISIC 2018 Task 1 | 2,594 | — | **Pixel-level segmentation masks** | TIxAI ground truth |
| PAD-UFES-20 | 2,298 | 6 | **Has Fitzpatrick skin type labels** | Fairness validation |
| Fitzpatrick17k | 16,577 | 114 conditions | FST labels, diverse skin tones | Extended fairness |

### Class Distribution in HAM10000

| Class | Full Name | Count | % |
|---|---|---|---|
| NV | Melanocytic nevi | 6,705 | 66.9% |
| MEL | Melanoma | 1,113 | 11.1% |
| BKL | Benign keratosis | 1,099 | 11.0% |
| BCC | Basal cell carcinoma | 514 | 5.1% |
| AKIEC | Actinic keratosis | 327 | 3.3% |
| DF | Dermatofibroma | 115 | 1.1% |
| VASC | Vascular lesions | 142 | 1.4% |

**Critical note:** NV is 66.9% of the dataset. Any global metric is dominated by NV. This is why per-class decomposition is necessary.

---

## 4. Saturated Components — What Is NOT Novel

These components must appear in Methods with citations to original authors. They must NEVER be claimed as contributions.

### Backbone & Training

| Component | Novelty | Kill Paper | Year |
|---|---|---|---|
| EfficientNetV2-S backbone | ❌ 1/10 | Tan & Le, ICML | 2021 |
| DenseNet121 | ❌ 1/10 | Huang et al., CVPR | 2017 |
| ConvNeXt-Tiny | ❌ 1/10 | Liu et al., CVPR | 2022 |
| Focal Loss | ❌ 1/10 | Lin et al., ICCV | 2017 |
| AdamW + cosine LR | ❌ 1/10 | Loshchilov & Hutter, ICLR | 2019 |
| Cross-attention fusion | ❌ 3/10 | Mridha & Islam, medRxiv | 2026 |

### XAI Methods

| Component | Novelty | Kill Paper | Year |
|---|---|---|---|
| Grad-CAM++ | ❌ 1/10 | Chattopadhyay et al., WACV | 2018 |
| SHAP DeepExplainer | ❌ 2/10 | Lundberg & Lee, NeurIPS | 2017 |
| Global TIxAI (aggregate) | ❌ 2/10 | Ieracitano et al., Neurocomputing | 2025 |
| MC Dropout uncertainty | ❌ 2/10 | Gal & Ghahramani, ICML | 2016 |
| Temperature scaling (ECE) | ❌ 1/10 | Guo et al., ICML | 2017 |

### Fairness & Deployment

| Component | Novelty | Kill Paper | Year |
|---|---|---|---|
| Per-FST accuracy fairness | ❌ 2/10 | Groh et al., CVPR | 2021 |
| ECE per skin tone | ❌ 2/10 | Bhattacharyya et al., Health Inf Sci Syst | 2026 |
| Conformal prediction fairness | ❌ 2/10 | Bhattacharyya et al., Health Inf Sci Syst | 2026 |
| ONNX INT8 + fairness + XAI on edge | ❌ 1/10 | SkinGuardian, Kumar & Dalwai, Research Square | 2026 |
| Composite Clinical Risk Score (CCRS) | ⚠️ 3/10 | Geifman & El-Yaniv, NeurIPS | 2017 |

**CCRS verdict:** Derivative — weighted combination of known outputs. Demote to engineering component. Never claim as headline contribution.

---

## 5. Forensic Novelty Audit — 8 Stages

The audit was applied to all 10+ claimed contributions. Rule at every stage: *search for the exact mechanism, not just the same topic. A paper must implement it to count as a kill. Verify every citation is real.*

| Stage | Test Question | Killed |
|---|---|---|
| 1 — Backbone check | Is the architecture novel? | EfficientNetV2-S, DenseNet121 |
| 2 — Loss & optimiser check | Are Focal Loss / AdamW / cosine LR novel? | All three |
| 3 — XAI method check | Are Grad-CAM++, SHAP, MC Dropout novel? | All three |
| 4 — Aggregate trust index check | Does a global inside-vs-outside-lesion metric already exist? | Global TIxAI (Ieracitano 2025) |
| 5 — Fairness & calibration check | Is per-FST accuracy or ECE already done? | Both (Bhattacharyya 2026, Groh 2021) |
| 6 — Edge deployment check | Is ONNX INT8 + fairness + XAI already combined? | Yes (SkinGuardian 2026) |
| 7 — Composite score check | Is a weighted deferral score novel? | CCRS demoted to engineering |
| 8 — Fine-grained XAI check | Per-class TIxAI / XAI disagreement deferral / failure prediction done? | **None found — all four survive** |

---

## 6. Four Surviving Novel Contributions

### Contribution 1 — Per-Class TIxAI

**Gap:** Ieracitano et al. 2025 define TIxAI as a global aggregate pooled across all images. Nobody stratifies explanation reliability by diagnostic class.

**Formula:**
```
Global TIxAI  = Σ[CAM(p)·M(p)] / (Σ[CAM(p)] + ε)   ← pooled, all classes, one number
Per-class TIxAI(c) = median{ TIxAI(i) : y_i = c AND ŷ_i = c }  ← per class, with 95% BCa CI
```

**Expected finding:** DF and VASC score 0.34–0.38 while NV scores 0.79 — a 2× reliability gap invisible to global TIxAI.

**Statistical validation:** Kruskal-Wallis across 7 classes + Holm-Bonferroni-corrected pairwise Mann-Whitney.

**Clinical meaning:** When the model diagnoses DF, its heatmap is unreliable. The clinician should not trust the highlighted region. This is actionable — and prior work never surfaced it.

**Novelty score: 8/10**

---

### Contribution 2 — Confusion Saliency Matrix (CSM)

**Gap:** Existing work shows saliency maps for individual images. No paper aggregates them by confusion pair (true → predicted class) to reveal systematic error-attention patterns.

**Construction:**
```
Step 1: Collect only misclassified samples
Step 2: Group by (true_class, pred_class) pair
Step 3: Apply n ≥ 5 guard (statistical stability)
Step 4: Average normalised Grad-CAM++ maps per cell
Step 5: Visualise as 7×7 grid; diagonal = correct (blank)
```

**Key finding:** Some confusions are "intelligent" (model attends inside lesion — genuine ambiguity). Others are "spurious" (model attends to hair/ruler/background — fixable with preprocessing). Same accuracy number, completely different clinical implications.

**Quantification:** Mean lesion-overlap ratio per cell = per-class TIxAI applied to error pairs. Clinician rating (Cohen's Kappa ≥ 0.60 required).

**Novelty score: 8/10**

---

### Contribution 3 — Multi-XAI Disagreement Index (XDI)

**Gap:** Selective prediction literature uses confidence or uncertainty as deferral signals. Nobody uses *disagreement between two XAI methods* as a deferral trigger in medical imaging.

**Formula:**
```
XDI = 1 − IoU(B_gradcam, B_shap)

where B_x = binary mask of top-k% activation/attribution
IoU = intersection / (union + ε)

XDI = 0  →  Grad-CAM++ and SHAP agree  →  Accept
XDI = 1  →  Complete disagreement       →  Defer to clinician
```

**Deferral policy:**
```
if XDI(x) > τ  →  defer to clinician
else            →  accept model output
τ chosen on validation set to hit target coverage (e.g. 80%)
```

**Critical ablation:** XDI-based selective accuracy curve vs. MC-Dropout-entropy-only curve at matched coverage. XDI must outperform uncertainty-alone to justify the contribution.

**Robustness:** Run at binarisation thresholds k = 25%, 50%, 75%. Mann-Whitney separation must hold at all three.

**Novelty score: 8/10**

---

### Contribution 4 — Explanation-Failure Prediction

**Gap:** No paper predicts whether a heatmap will be unreliable *before* generating it. This enables real-time flagging without running the full XAI pipeline.

**Meta-model feature vector:**
```python
features = {
    'max_confidence':     softmax.max(),
    'confidence_margin':  softmax[top1] - softmax[top2],
    'entropy':            -sum(p * log(p) for p in softmax),
    'mc_mean_confidence': mc_probs.mean(),
    'mc_variance':        mc_probs.var(),
    'predictive_entropy': entropy(mc_probs.mean(axis=0)),
    'pca_embedding':      pca.transform(embedding)[0]   # 32-dim
}
```

**Label:** `y_meta = (TIxAI(i) < 0.45).astype(int)`  — 1 = "explanation will fail"

**Training (leakage-free):**
```python
meta_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=32)),
    ('clf',    GradientBoostingClassifier(n_estimators=200))
])
y_pred_proba = cross_val_predict(meta_model, X, y_meta, cv=5, method='predict_proba')
```

**Target:** AUC ≥ 0.70 with 95% BCa CI lower bound > 0.65.

**Novelty score: 8/10**

---

## 7. Existing Literature — Paper-wise Analysis

| # | Paper | Year | Venue | Model | Dataset | XAI | Uncertainty | Key Limitation |
|---|---|---|---|---|---|---|---|---|
| 1 | Mahmud et al. | 2025 | Scientific Reports | Custom CNN (GAP, BN, Swish) | Kaggle Melanoma | Grad-CAM, Saliency | None | No dark skin; no uncertainty; no ERI |
| 2 | Murali et al. (DermaScanAI) | 2026 | Scientific Reports | ConvNeXt-Tiny + Dual Attention + SE | HAM10000 | Grad-CAM++, SHAP | None | HAM10000 only; no metadata; compute-heavy |
| 3 | Ahmad et al. | 2023 | Frontiers in Oncology | Xception + ShuffleNet + Butterfly Optim. | HAM10000 + ISIC 2018 | Grad-CAM | None | No uncertainty; no calibration |
| 4 | Syed et al. | 2024 | Journal of Imaging | Xception + PSO + SVM/KNN | ISIC 2018, HAM10000 | Grad-CAM, LIME, Occlusion | None | Binary only; no FST fairness |
| 5 | Mukherjee et al. | 2026 | Discover Oncology | Custom DCNN + Swish | HAM10000 | LIME + Grad-CAM | None | Single dataset; light-skin bias acknowledged not solved |
| 6 | Asgharnezhad et al. | 2025 | arXiv | ResNet50, DenseNet121, EfficientNetV2, CLIP ViT | HAM10000 | None | MC Dropout + Ensembles | No XAI at all; no FST analysis |
| 7 | Naz et al. | 2025 | Scientific Reports | VGG19 FedAvg (25 rounds) | ISIC 2017 | Grad-CAM | None | Binary task; no uncertainty |
| 8 | Groh et al. | 2021 | CVPR | ResNet + Clinical | Fitzpatrick17k | None | None | No quantitative XAI metric |
| 9 | Ieracitano et al. | 2025 | Neurocomputing | CNN + attention | HAM10000 | TIxAI (global) | None | Global only — no per-class stratification |
| 10 | Bhattacharyya et al. | 2026 | Health Inf Sci Syst | ViT foundation model | HAM10000 + PAD-UFES | None | Conformal prediction | No XAI; no explanation reliability |
| 11 | SkinGuardian (Kumar & Dalwai) | 2026 | Research Square (preprint) | BEiT ViT | Fitzpatrick17k | Grad-CAM | DP-SGD | No per-class TIxAI; no CSM; no XDI |
| 12 | Daneshjou et al. | 2022 | Science Advances | Multiple | Fitzpatrick17k, DDI | None | None | Prediction fairness only — not explanation fairness |
| 13 | Mridha & Islam | 2026 | medRxiv (preprint) | Metadata cross-attention | PAD-UFES-20 | None | None | No TIxAI; no uncertainty |
| 14 | DeepMetaForge (Vachmanus et al.) | 2023 | IEEE Access | ViT + metadata | HAM10000 | SHAP | None | No class-level reliability; no deferral |
| 15 | Pampana & Mazumder | 2026 | Scientific Reports | Custom CNN | HAM10000 | Grad-CAM++ | None | Qualitative mask consistency only |

---

## 8. Research Gap Analysis

### Technical Gaps

| Gap | Evidence | Why Unsolved |
|---|---|---|
| Per-class explanation reliability not measured | Ieracitano 2025 gives only global TIxAI | Global metric masks rare-class failures |
| XAI disagreement as deferral signal | No medical imaging paper found | Deferral literature uses confidence/uncertainty only |
| Explanation-failure prediction before generation | No paper found | Computational XAI assumes generation is always run |
| Confusion-pair saliency patterns | Young et al. 2019 shows individual anecdotes | Nobody aggregates systematically by error type |

### Dataset Gaps

| Gap | Evidence | Impact |
|---|---|---|
| HAM10000 has no FST labels | Daneshjou et al. 2022 | Cannot do per-class × FST interaction analysis |
| Rare classes (DF=115, VASC=142) too small | HAM10000 class distribution | Per-class statistics have wide CIs |
| ISIC 2018 masks not always attached | Notebook warns of synthetic-mask contamination | TIxAI results invalid without real masks |

### Explainability Gaps

| Gap | Evidence | Current State |
|---|---|---|
| No quantitative explanation reliability per class | All papers visual-only | ERI exists globally (Ieracitano 2025) but not per class |
| No XAI method comparison using quantitative metric | All papers choose one method arbitrarily | Comparison is anecdotal |
| No pre-generation failure prediction | No paper found | First-in-literature opportunity |
| No systematic confusion-pair analysis | No paper found | First-in-literature opportunity |

### Fairness Gaps

| Gap | Evidence | Current State |
|---|---|---|
| Explanation quality fairness by FST | Daneshjou 2022 measures prediction fairness | Nobody measures if heatmaps are equally reliable across skin tones |
| Per-class × FST interaction | No paper found | Three-way interaction (class × FST × TIxAI) unexplored |

### Clinical Validation Gaps

| Gap | Evidence | Impact |
|---|---|---|
| No clinician rating of heatmaps at scale | Chanda et al. 2025 shows 93.6% agreement (but for Grad-CAM not for reliability scoring) | Cannot claim clinical usability without reader study |
| No prospective deployment study | All papers use retrospective public datasets | Limits journal target |

### Computational Efficiency Gaps

| Gap | Evidence | Impact |
|---|---|---|
| Running full XAI pipeline is expensive | — | Failure prediction (contribution 4) addresses this directly |

---

## 9. Comparison Table

| Paper | Dataset | Model | XAI | Accuracy | Key Limitation | Future Work Suggested |
|---|---|---|---|---|---|---|
| Mahmud 2025 | Kaggle Melanoma | Custom CNN | Grad-CAM | 96.5% | No FST, no uncertainty | Dark skin validation |
| DermaScanAI 2026 | HAM10000 | ConvNeXt-Tiny | Grad-CAM++, SHAP | ~90%+ | No metadata, no uncertainty | Metadata fusion |
| Ahmad 2023 | HAM10000+ISIC | Xception+ShuffleNet | Grad-CAM | 99.3%/91.5% | No uncertainty, no calibration | Uncertainty addition |
| Syed 2024 | ISIC 2018, HAM10000 | Xception+PSO | Grad-CAM, LIME | 98.5%/86.1% | Binary only | Multi-class |
| Mukherjee 2026 | HAM10000 | DCNN+Swish | LIME+Grad-CAM | 98.31% | Light-skin bias | Dark-skin data |
| Asgharnezhad 2025 | HAM10000 | EfficientNetV2, ViT | None | — | No XAI at all | XAI addition |
| Groh 2021 | Fitzpatrick17k | ResNet+Clinical | None | FST-stratified | No XAI | Explanation fairness |
| Ieracitano 2025 | HAM10000 | CNN+attention | TIxAI (global) | — | Global only | Per-class TIxAI |
| Bhattacharyya 2026 | HAM10000+PAD | ViT | None | Conformal fairness | No XAI | Explanation integration |
| SkinGuardian 2026 | Fitzpatrick17k | BEiT ViT | Grad-CAM | — | No per-class; no XDI | Per-class analysis |
| **FairSkinNet (ours)** | **HAM10000+PAD-UFES** | **EfficientNetV2-S+Attn** | **Grad-CAM++, SHAP, TIxAI(c), XDI, CSM** | **[TODO: run]** | **External validation pending** | **Clinician study** |

---

## 10. Common Problems Across Literature

Ranked from most to least critical:

| Rank | Problem | Papers Reporting | Frequency | Our Solution |
|---|---|---|---|---|
| 1 | Visual-only XAI — no quantitative reliability metric | All 9 XAI papers | 100% | Per-Class TIxAI |
| 2 | Single dataset (HAM10000) | 7/9 papers | 78% | HAM10000 + PAD-UFES-20 + ISIC 2018 |
| 3 | No fairness measurement across skin tones | 8/9 papers | 89% | Per-FST stratification |
| 4 | Class imbalance not addressed beyond reweighting | All papers | 100% | Focal Loss + class-weighted sampler |
| 5 | XAI and uncertainty never combined | All papers separate | 100% | MC Dropout + TIxAI correlation |
| 6 | No external validation | 7/9 papers | 78% | PAD-UFES-20 as external set |
| 7 | Explanation stability under perturbation not tested | All papers | 100% | XDI robustness at k=25,50,75% |
| 8 | No deferral policy | All papers | 100% | XDI-based deferral + failure predictor |
| 9 | No clinician validation | All papers | 100% | Reader study (planned) |
| 10 | Light-skin dataset bias | Mukherjee 2026, Groh 2021 | 22% documented, 89% implicit | PAD-UFES-20 FST metadata |

---

## 11. Notebook Code Review — What Needs Fixing

### Critical Bugs (Block Submission Until Fixed)

| Issue | Location | Fix |
|---|---|---|
| **Synthetic mask contamination** | Cell 10 — TIxAI computation | Re-run with real ISIC 2018 segmentation masks. Confirm `TIXAI_USES_SYNTHETIC_MASKS = False`. Zero tolerance for contaminated results in paper. |
| **Data leakage in meta-model** | Cell 14 — failure predictor | `scaler.fit_transform(X)` and `PCA.fit_transform(emb)` run on full dataset before CV. Wrap both inside `sklearn.Pipeline`. Current AUC is inflated. |
| **TIxAI threshold not validated** | Cell 14 — label definition | `y_meta = (TIxAI < 0.45)` uses a fixed arbitrary cutoff. Run sensitivity analysis at 0.35, 0.45, median TIxAI. Report AUC at all three. |

### Important Issues (Fix Before Submission)

| Issue | Location | Fix |
|---|---|---|
| XDI binarisation at single threshold | Cell 12 | Run Mann-Whitney at k = 25%, 50%, 75%. Report all three. One-threshold result will be questioned. |
| ConvNeXt baseline may be undertrained | Cell 8 | Confirm DenseNet121 and ConvNeXt get identical epochs, augmentation, and early-stopping criteria. |
| Multiple comparisons not corrected | Cell 11 | Confirm Holm-Bonferroni applied to all 7-class pairwise Mann-Whitney tests. |
| No external validation set | Full notebook | Add PAD-UFES-20 or BCN20000 as a second test set for all four contributions. |

### Scores (Current vs. After Fixes)

| Dimension | Current | After Fixes |
|---|---|---|
| Novelty | 8/10 | 8/10 |
| Mathematical correctness | 6/10 | 8.5/10 |
| Data integrity | 4.5/10 | 9/10 |
| Engineering quality | 7.5/10 | 8.5/10 |
| Publication readiness | 5/10 | 8.5/10 |

---

## 12. Validation Experiments & Statistical Tests

### Contribution 1 — Per-Class TIxAI

| Experiment | Method | Statistical Test | Output |
|---|---|---|---|
| Main per-class scores | Median TIxAI per class | — | Table: class × median × CI |
| Significance of class differences | Kruskal-Wallis (7 classes) | H-statistic, p-value | Confirms classes differ |
| Post-hoc pairwise | Mann-Whitney + Holm-Bonferroni | p-adjusted per pair | Identifies which pairs differ |
| Real mask confirmation | Re-run with ISIC 2018 masks only | — | Kills synthetic-mask objection |
| External validation | Re-run on PAD-UFES-20 | Kruskal-Wallis | Generalisability |
| FST stratification | Per-class TIxAI by FST group | Mann-Whitney (light vs dark) | Fairness extension |

**Minimum bar:** Kruskal-Wallis p < 0.05, ≥1 pairwise contrast surviving Holm-Bonferroni, zero synthetic masks.

### Contribution 2 — Confusion Saliency Matrix

| Experiment | Method | Statistical Test | Output |
|---|---|---|---|
| Main CSM | Aggregated Grad-CAM++ per confusion pair | n ≥ 5 guard | 7×7 visual grid |
| Lesion overlap per cell | Mean TIxAI on error-pair maps | — | Intelligent vs spurious classification |
| Clinician rating | 5–10 dermatologists rate 20–30 cells | Cohen's Kappa | Clinical validity |
| Stability check | 2 random seeds | — | Cell averages stable |

**Minimum bar:** ≥6 valid cells, Kappa ≥ 0.60.

### Contribution 3 — XDI Deferral

| Experiment | Method | Statistical Test | Output |
|---|---|---|---|
| Separation test | XDI distribution: correct vs incorrect | Mann-Whitney U, one-sided | Core claim |
| Effect size | Rank-biserial r | — | p-value alone insufficient |
| Selective accuracy curve | Accuracy vs coverage at τ = 10th–90th percentile | — | Shows deferral works |
| Head-to-head baseline | Same curve for MC-Dropout entropy | — | Proves XDI adds value |
| Threshold robustness | Mann-Whitney at k = 25%, 50%, 75% | — | Not threshold-sensitive |
| AUSC metric | Area under selective accuracy curve | — | Single summary number |

**Minimum bar:** Mann-Whitney p < 0.01, r ≥ 0.15, XDI curve above uncertainty-only at coverage = 60%, 70%, 80%.

### Contribution 4 — Explanation-Failure Prediction

| Experiment | Method | Statistical Test | Output |
|---|---|---|---|
| Main AUC | 5-fold CV inside Pipeline (leakage-free) | — | AUC + 95% BCa CI |
| Bootstrap CI | 2000 resamples, BCa method | — | Medical AI standard |
| Ablation table | Confidence-only / +MC entropy / +PCA / full | — | Each feature group contributes |
| Threshold sensitivity | Re-run at TIxAI cutoff 0.35, 0.45, median | — | Label definition doesn't drive result |
| Meta-model calibration | Reliability diagram + ECE | — | Meta-model itself must be calibrated |

**Minimum bar:** AUC ≥ 0.70, CI lower bound > 0.65, embedding features add lift over uncertainty-only.

### Universal Statistical Requirements

```
1. Effect sizes with every p-value (Cohen's d, rank-biserial r, partial η²)
2. Holm-Bonferroni correction on all multi-class pairwise tests
3. Non-parametric tests throughout (TIxAI is bounded [0,1], likely skewed)
4. BCa bootstrap CIs on every headline number (2000 resamples)
5. Fixed random seeds — report seed values in paper
```

---

## 13. Novel Research Opportunities

Ranked by novelty × feasibility:

| Rank | Idea | Novelty | Feasibility | Why Novel | Publication Target |
|---|---|---|---|---|---|
| 1 | **Explanation quality fairness by FST** — ERI stratified by Fitzpatrick | 9/10 | HIGH | Daneshjou 2022 measures prediction fairness; nobody measures explanation fairness | Computers in Biology and Medicine |
| 2 | **Explanation drift early warning (EDEWS)** — rolling TIxAI + CUSUM/ADWIN to detect model degradation before accuracy drops | 9.5/10 | MEDIUM | MMC+ 2024 monitors prediction drift; zero papers monitor saliency spatial quality drift | Nature Machine Intelligence / IEEE TMI |
| 3 | **Explanation calibration curves (ERCC)** — isotonic regression mapping confidence → expected TIxAI | 9/10 | HIGH | Lofstrom 2024 calibrates SHAP weights not spatial quality | Medical Image Analysis |
| 4 | **Uncertainty stratified by FST** — is MC Dropout entropy higher for darker skin? | 8.5/10 | VERY HIGH | BOSQUE 2025 discusses epistemic uncertainty in fairness but does NOT stratify per-image entropy by FST | Computers in Biology and Medicine |
| 5 | **XDI vs uncertainty deferral comparison** (partially in paper already) | 8/10 | HIGH | Selective prediction uses confidence; XAI-disagreement as signal is new | MICCAI 2027 |
| 6 | **Three-way fairness audit (class × FST × TIxAI)** | 8.5/10 | MEDIUM | No paper does three-way interaction | IEEE JBHI |
| 7 | **Continuous ITA-ERI regression** — skin melanin index vs explanation quality | 9/10 | HIGH | Binary FST grouping is coarse; continuous ITA more granular | Expert Systems with Applications |
| 8 | **Uncertainty-gated explanation filter** — suppress heatmaps with entropy > τ AND TIxAI < θ | 8/10 | HIGH | Clinical deployment-ready; no prior operational filter system | Biomedical Signal Processing and Control |
| 9 | **Conformal prediction intervals for TIxAI** — guaranteed coverage intervals for explanation reliability | 8/10 | MEDIUM | Conformal prediction rising in medical AI; not applied to spatial saliency | arXiv → journal |
| 10 | **Per-class explanation fairness score** — TIxAI × FST × class three-way interaction | 8/10 | MEDIUM | No three-way analysis found | Q1 journal |
| 11 | **Explanation stability under clinical augmentations** — IoU across rotation/brightness/JPEG perturbations by FST | 7.5/10 | HIGH | Vannucci 2026 measures sensitivity to optical parameters but not by FST | Computers in Biology and Medicine |
| 12 | **Skin-tone invariant explanation calibrator** — post-hoc recalibration producing equal-quality heatmaps across FST | 9.5/10 | MEDIUM | FairDisCo removes FST from features (prediction) — not explanation quality | CVPR Medical track |
| 13 | **Explanation entropy as uncertainty proxy** — spatial entropy of heatmap as second uncertainty signal | 7/10 | HIGH | Separate from predictive entropy; spatial disorder of attention not studied | Q2 journal |
| 14 | **XAI-weighted ensemble** — weight model predictions by explanation quality, not confidence | 7.5/10 | MEDIUM | Ensemble weighting by confidence is standard; by explanation quality is new | Q1 journal |
| 15 | **Human-in-the-loop explanation feedback** — dermatologist corrections to Grad-CAM used as additional loss | 8/10 | LOW | Active XAI learning not in dermatology | JAMIA / HCI venue |

---

## 14. Research Gap Matrix

| Gap | Existing Papers | Why Unsolved | Proposed Solution |
|---|---|---|---|
| Per-class explanation reliability | Ieracitano 2025 (global only) | Global metric masks rare-class failures | Per-Class TIxAI with BCa CI per class |
| Systematic confusion-pair saliency | Young et al. 2019 (per-image anecdotes) | No aggregation framework exists | Confusion Saliency Matrix with n≥5 guard |
| XAI disagreement as deferral signal | Geifman & El-Yaniv 2017 (confidence-based) | Deferral literature uses confidence/uncertainty only | XDI = 1 − IoU(Grad-CAM++, SHAP) + policy |
| Pre-generation failure prediction | None in medical XAI | Assumes XAI pipeline always runs | Meta-model predicting low TIxAI from confidence + entropy + embedding |
| Explanation quality fairness by FST | Groh 2021, Bhattacharyya 2026 (prediction fairness only) | No paper extends fairness to explanation quality | Per-class TIxAI × FST stratification |
| Explanation drift early warning | MMC+ 2024 (prediction-level drift) | XAI quality never monitored over time | Rolling ERI + CUSUM/ADWIN on TIxAI stream |
| Explanation calibration curves | Lofstrom 2024 (SHAP weights) | Spatial saliency calibration never attempted | Isotonic regression: confidence → expected TIxAI |
| Clinician validation of explanation reliability | No systematic reader study in literature | Expensive; requires dermatologist access | Reader study: 5–10 dermatologists rate CSM cells |

---

## 15. Novelty Score Summary

| Contribution | Novelty | Scientific Impact | Publication Potential | Reviewer Risk | Status |
|---|---|---|---|---|---|
| EfficientNetV2-S backbone | 1/10 | Low | None | CRITICAL rejection | ❌ Do not claim |
| Grad-CAM++ alone | 1/10 | Low | None | CRITICAL rejection | ❌ Do not claim |
| MC Dropout alone | 2/10 | Low | None | HIGH rejection | ❌ Do not claim |
| Global TIxAI | 2/10 | Low | None | CRITICAL (exists 2025) | ❌ Do not claim |
| Per-FST accuracy | 2/10 | Low | None | HIGH (Groh 2021) | ❌ Do not claim |
| ONNX INT8 edge | 1/10 | Low | None | HIGH (SkinGuardian) | ❌ Do not claim |
| CCRS composite | 3/10 | Low | Engineering only | MEDIUM | ⚠️ Engineering section only |
| **Per-Class TIxAI** | **8/10** | **HIGH** | **Medical Image Analysis / IEEE TMI** | Low if real masks used | ✅ Core contribution |
| **Confusion Saliency Matrix** | **8/10** | **HIGH** | **MICCAI / Computers Bio Med** | Low if clinician rating included | ✅ Core contribution |
| **XDI Deferral** | **8/10** | **HIGH** | **MICCAI / IEEE TMI** | Low if deferral baseline shown | ✅ Core contribution |
| **Failure Prediction** | **8/10** | **HIGH** | **Medical Image Analysis** | Medium — needs leakage fix | ✅ Core contribution (fix bug first) |

---

## 16. Reviewer Perspective (Reviewer #2)

### As MICCAI 2027 Reviewer

**Strengths:**
- Four technically distinct contributions with clear operationalisation
- Kruskal-Wallis + Holm-Bonferroni is appropriate for non-parametric multi-class comparison
- n ≥ 5 guard on CSM shows statistical awareness
- Pipeline (scaler+PCA inside CV) shows methodological care (after fix)

**Major weaknesses:**
1. *"Results reported with synthetic masks are not valid. The paper must confirm all TIxAI numbers use real ISIC 2018 ground-truth masks."*
2. *"There is no external validation. All four contributions are measured only on HAM10000. They could be dataset-specific artifacts."*
3. *"The failure predictor reports AUC from a known-leaky pipeline. The corrected AUC must be reported."*
4. *"The deferral policy comparison vs. uncertainty-only baseline is missing. Without this, XDI is not justified over simpler alternatives."*

**Questions:**
- How sensitive are per-class TIxAI scores to the TIxAI threshold (0.45)?
- What is Cohen's Kappa on the clinician rating of CSM cells?
- Is the XDI Mann-Whitney result stable across k = 25%, 50%, 75%?
- What is the corrected (leakage-free) failure predictor AUC?

**Recommendation:** Borderline → Accept after major revision (fix masks + leakage + external set + deferral baseline)

---

### As Medical Image Analysis Reviewer

**Strengths:**
- Accuracy-vs-trust dissociation as headline finding is clinically meaningful and novel framing
- Four contributions are internally consistent — they all address the same thesis from different angles
- Per-class stratification directly actionable in clinical deployment

**Major weaknesses:**
1. *"No clinician validation. A purely computational claim about 'clinical trustworthiness' requires at least a small reader study."*
2. *"PAD-UFES-20 is smaller and uses smartphone cameras, not dermoscopes. Generalisability to clinical dermoscopy settings is not established."*
3. *"The CSM is visually compelling but quantification is weak. Mean lesion-overlap per cell needs statistical testing between intelligent and spurious categories."*

**Recommendation:** Weak Accept (after minor revision + clinician check)

---

## 17. Publication Strategy

### Primary Target
**Computers in Biology and Medicine** (Elsevier)
- IF: ~7.7 | CiteScore: ~12
- Perfect scope: ML for medical imaging, fairness in medical AI, XAI
- Typical accept rate: 25–30% for structured methodology papers
- Estimated time: 8–12 weeks first decision

### Secondary Target
**Medical Image Analysis** (Elsevier)
- IF: ~10.7 | More prestigious, stricter
- Requires external validation + clinical component
- Suitable after PAD-UFES-20 external validation added

### Conference Target (Parallel)
**MICCAI 2027** — 8-page workshop paper presenting CSM + XDI together
- Builds citation base before journal submission

### Avoid
- Lancet Digital Health, JAMIA: require clinical trial data
- IEEE TPAMI: requires algorithmic novelty beyond medical application

### Submission Readiness Checklist

| Gate | Status | Action |
|---|---|---|
| Real ISIC 2018 masks | ❌ Pending | Re-run notebook with real masks |
| Leakage bug fixed | ❌ Pending | Wrap scaler+PCA in Pipeline |
| External test set | ❌ Pending | PAD-UFES-20 or BCN20000 results |
| XDI vs uncertainty deferral baseline | ❌ Pending | Run selective accuracy comparison |
| Clinician reader study | ❌ Pending | 5–10 dermatologists, 20–30 CSM cells |
| Multiple-comparisons correction | ⚠️ Partial | Confirm Holm-Bonferroni in Cell 11 |
| TIxAI threshold sensitivity | ❌ Pending | Run at 0.35, 0.45, median |
| All seeds fixed | ⚠️ Partial | Confirm random_state in all cells |

---

## 18. Final Research Roadmap

### Existing State (What We Have)
- Full FairSkinNet notebook (2,588 lines, 19 cells) with working implementation of all four contributions
- 8-stage forensic novelty audit completed — 4 contributions confirmed open
- Literature survey of 15 papers (2021–2026) completed
- Master paper prompt written and ready

### What Must Be Done Before Submission (Ordered)

| Priority | Task | Estimated Time | Blocks |
|---|---|---|---|
| P0 | Attach real ISIC 2018 masks; re-run Cell 10 | 1 day | Everything |
| P0 | Fix scaler+PCA leakage in Cell 14 | 2 hours | Failure predictor AUC |
| P1 | Run TIxAI threshold sensitivity (0.35, 0.45, median) | 2 hours | Contribution 4 robustness |
| P1 | Run XDI at k=25%, 50%, 75% | 2 hours | Contribution 3 robustness |
| P1 | Add PAD-UFES-20 as external test set | 2 days | External validation |
| P2 | Run XDI-deferral vs MC-Dropout-entropy comparison | 3 hours | Contribution 3 ablation |
| P2 | Reader study (5–10 dermatologists on CSM) | 1–2 weeks | Contribution 2 validation |
| P3 | Write manuscript using master prompt | 1 week | Publication |

### Recommended Strongest Research Direction

**Single-sentence thesis:**
> Accuracy and explanation trustworthiness are dissociable, class-dependent, and predictable in advance — and FairSkinNet's four-part framework measures, visualises, detects, and predicts this dissociation.

**Recommended paper structure:**
1. Abstract — accuracy-vs-trust dissociation as hook
2. Introduction — gap in per-class explainability reliability
3. Related Work — TIxAI 2025, SkinGuardian 2026, Bhattacharyya 2026, Groh 2021
4. Methods — backbone (brief, non-novel) + four contributions in full
5. Experiments — per-contribution results with statistical tests
6. Ablations — mask sensitivity, threshold sensitivity, deferral baseline
7. Discussion — clinical implications, limitations (explicit subsection)
8. Conclusion — restate thesis, four contributions

---

## 19. Master Summary Table

| Research Gap | Existing Solution | Limitation of Existing | Future Work | Our Opportunity | Novelty | Publication Potential | Recommended Next Step |
|---|---|---|---|---|---|---|---|
| Per-class explanation reliability | Global TIxAI (Ieracitano 2025) | Single aggregate, rare classes invisible | Per-class decomposition | Per-Class TIxAI with BCa CI | 8/10 | Medical Image Analysis / Computers Bio Med | Real masks → re-run → submit |
| Systematic confusion-pair saliency | Per-image anecdotes (Young 2019) | Not aggregated by error type | Aggregated error-pattern maps | Confusion Saliency Matrix | 8/10 | MICCAI / Computers Bio Med | Reader study (5–10 dermatologists) |
| XAI disagreement as deferral signal | Confidence/uncertainty deferral (Geifman 2017) | Uses confidence not explanation | XAI-aware deferral | XDI = 1 − IoU(Grad-CAM++, SHAP) | 8/10 | MICCAI / IEEE TMI | Add deferral baseline comparison |
| Pre-generation failure prediction | None | XAI pipeline always runs | Pre-generation flag | Meta-model predicting low TIxAI | 8/10 | Medical Image Analysis | Fix leakage bug → re-run |
| Explanation fairness by FST | Prediction fairness (Groh 2021, Bhattacharyya 2026) | Prediction only, not explanation | Explanation fairness audit | Per-class TIxAI × FST | 9/10 | Computers Bio Med / npj Digital Medicine | Add PAD-UFES-20 FST stratification |
| Explanation drift detection | Prediction drift monitoring (MMC+ 2024) | Does not monitor XAI quality | Saliency drift as early warning | Rolling TIxAI + CUSUM | 9.5/10 | Nature Machine Intelligence / IEEE TMI | Future PhD chapter / separate paper |
| Explanation calibration curves | Probability calibration (Guo 2017) | Calibrates confidence not saliency | Saliency reliability curves | Isotonic regression: conf → TIxAI | 9/10 | Medical Image Analysis | Future paper |

---

*Report compiled from: TrustXAI_Derm_Kaggle_Notebook.md, FairSkinNet_Novelty_Destruction_Report.md, FairSkinNet_Research_Gap_Novelty_Blueprint.md, FairSkinNet_Complete_Experiment_Guide.md, FairSkinNet_MaximumNovelty_10Ideas.md, deep-research-report.md, deep-research-report1.md, deezp-research-report.md, TrustXAI_Derm_Unified_Paper_Blueprint.md, TrustXAI_Derm_Master_Paper_Prompt.md*

*Evidence-based — every claim traceable to an uploaded document or a verified external paper. No fabricated citations.*
