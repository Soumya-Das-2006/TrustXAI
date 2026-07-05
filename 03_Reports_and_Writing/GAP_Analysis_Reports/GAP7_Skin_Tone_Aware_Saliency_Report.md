# GAP #7 — SKIN-TONE AWARE SALIENCY
# Auditing Explanation Fairness Across Skin Tones
## Ultra-Deep Research & Design Report

### **When Should Clinicians Not Trust an AI Explanation?**
> A Trust-Aware Framework for Dermoscopic Skin Cancer Diagnosis  
> **Contribution 5: Skin-Tone Aware Saliency (Explanation Fairness)**

---

**Document Class:** Q1 Journal Research Design Document  
**Target Venues:** Medical Image Analysis · Nature Digital Medicine · IEEE Transactions on Medical Imaging · Computers in Biology and Medicine · Expert Systems with Applications  
**Research Team:** Senior Machine Learning Engineer · Computer Vision Research Scientist · Explainable AI (XAI) Researcher · Medical AI Researcher · Fairness in AI Researcher · Dermatology Specialist · Clinical AI Safety Researcher · Medical Imaging Scientist · Statistician · Q1 Journal Reviewer (Reviewer #2)  
**Context:** Assumes the base classifier (DenseNet121, Gap #1), Grad-CAM++ (Gap #1), and baseline explanation metrics (TIxAI, Gap #1) are established. This study investigates whether deep learning models exhibit demographic disparities in explanation quality and visual attention across different human skin tones.  
**Version:** 1.0  
**Date:** June 2026

---

## TABLE OF CONTENTS

| # | Section |
|---|---------|
| 1 | The Scientific Question |
| 2 | Comprehensive Literature Review |
| 3 | Research Gap Identification |
| 4 | Dataset Feasibility Report |
| 5 | Model Requirements |
| 6 | Architecture Comparison & Selection |
| 7 | Complete ML Pipeline |
| 8 | Fairness Analysis Design |
| 9 | Dataset Strategy |
| 10 | Statistical Analysis Plan |
| 11 | Clinical Interpretation Framework |
| 12 | Reviewer #2 Simulation |
| 13 | Robustness and Reproducibility |
| A | Notation Glossary |
| B | Ethical & Clinical Considerations |

---

## SECTION 1 — THE SCIENTIFIC QUESTION

### 1.1 What Is Actually Being Asked?

The primary scientific inquiry of GAP #7 is not:

> *"Does the model achieve equal classification accuracy across light and dark skin tones?"*

That is a question of **accuracy fairness** (demographic parity or equalized odds in prediction outputs).

Nor is it:

> *"Can we use color normalization to improve model performance?"*

Instead, it addresses the following question:

> **Does the patient's skin tone influence the reliability and spatial focus of the AI's explanation, even when the model's prediction accuracy remains similar across subgroups?**

In clinical dermatology, AI diagnostic models are often trained on datasets consisting predominantly of Caucasian skin types (Fitzpatrick Skin Types I and II). When deployed on patients with darker skin tones (Fitzpatrick Skin Types V and VI), these models can fail in subtle, dangerous ways.

Even if a model correctly classifies a malignant lesion on a dark-skinned patient (e.g., achieving equal prediction accuracy), its **explanation map** (e.g., Grad-CAM++ activation) may focus on non-clinical features—such as healthy skin pigmentation patterns, hair structures, or illumination gradients—rather than the actual diagnostic characteristics of the lesion. This phenomenon, termed **explanation bias**, indicates that the model is making correct predictions for the wrong reasons in specific demographic subgroups.

```
+-------------------------------------------------------+
|                THE EXPLANATION FAIRNESS GAP           |
|                                                       |
|   Light Skin Patient (FST I-II):                      |
|   Label: Correct (MEL) ===> CAM Focus: Lesion Center  |
|   Reasoning: Trustworthy                              |
|                                                       |
|   Dark Skin Patient (FST V-VI):                       |
|   Label: Correct (MEL) ===> CAM Focus: Healthy Skin   |
|   Reasoning: Unreliable (Shortcut to background)      |
|   *Accuracy is identical; trust is fundamentally biased* |
+-------------------------------------------------------+
```

### 1.2 Why Explanation Fairness Matters Clinically

1.  **Automation Bias Safety Net:** If a dermatologist receives a correct label but notes that the AI highlighted normal dark skin features rather than the atypical pigment network, they will know to discount the AI's confidence. If the explanation quality is lower for darker skin tones, clinicians cannot trust the AI's diagnostic reasoning, increasing the risk of misdiagnosis for minority populations.
2.  **Mitigating Under-Diagnosis:** Melanin absorbs light, reducing contrast between skin lesions and background skin in darker skin tones. AI models can easily rely on skin-tone context rather than lesion boundaries, leading to false-positive or false-negative attributions.
3.  **Auditing the Decision Logic:** Demonstrating that explanations are fair ensures that the neural networks are learning representation spaces that are truly invariant to patient demographics.

### 1.3 ACTUAL RESULT ON HAM10000 (Post-Execution Update, 2026-07-05)

**Section 4 below correctly notes that HAM10000 has no clinically-assigned Fitzpatrick labels**, and that remains true. What this section adds is that the notebook ran a directional audit on HAM10000 anyway (CELL 10b / CELL 14c), using peri-lesional **ITA (Individual Typology Angle) as an estimated proxy** for Fitzpatrick skin type, bucketed into FST I-II / III-IV / V-VI. This is explicitly a proxy, not a substitute for the gold-standard MSKCC or PAD-UFES-20 data recommended in Section 4 — but since it did produce a real, reportable result, it belongs in this document rather than being left out because the "primary" dataset plan (Section 4.1.6) was never attached to this project.

**Two distinct findings came out of that proxy audit — they answer different questions and should not be conflated:**

1. **Does estimated skin tone predict TIxAI (explanation *localization* quality)?** — **No signal detected.**
   - FST I-II (n=352): median TIxAI = 0.456. FST III-IV (n=94): median TIxAI = 0.423. FST V-VI (n=20): median TIxAI = 0.398.
   - Kruskal-Wallis: **H = 3.216, p = 0.20 → FAIL TO REJECT H0.** No statistically significant difference in explanation localization quality across the estimated FST buckets.

2. **Does estimated skin tone predict the model's own predictive uncertainty (MC-Dropout entropy)?** — **Yes, a significant gap.**
   - FST I-II: mean entropy = 1.307 ± 0.336. FST III-IV: mean entropy = 1.418 ± 0.287. FST V-VI: mean entropy = 1.475 ± 0.331.
   - Kruskal-Wallis: **H = 15.634, p = 0.0004 → REJECT H0.** The model is systematically more uncertain about its own predictions for images estimated to be darker-skinned, even though the *localization quality* of its explanations (finding 1) does not measurably differ.

**Read together:** this is a "the explanation points to the right place about equally often, but the model is less sure of itself" pattern for the estimated darker-skin bucket — not the "explanation actively points at the wrong region" pattern the Section 1.1 hypothetical (light-skin patient CAM-on-lesion vs. dark-skin patient CAM-on-healthy-skin) illustrates. Report both findings, not just one; do not summarize this as "no fairness issue found" (finding 2 is a real fairness-relevant gap) or as "explanation bias confirmed" (finding 1 does not support that framing).

**Caveats that must travel with both numbers:**
- **ITA is an estimated proxy, not a clinically-assigned Fitzpatrick label.** HAM10000 has zero ground-truth skin-tone metadata (Section 4.1.1 is correct about this). Treat both findings as directional signals requiring validation against real FST labels (Fitzpatrick17k, PAD-UFES-20, or the MSKCC dataset in Section 4.1.6) before any clinical or regulatory claim is built on them.
- **FST V-VI has only n=20** in this sample — the widest-uncertainty, smallest-n bucket in both tests. Both p-values should be re-checked as more darker-skin-estimated samples accumulate; a single-digit swing in this bucket's classification could change significance in either test.
- Both results, like every other DenseNet121-dependent finding in this project, reflect a checkpoint trained to only 60/150 epochs under a since-fixed scheduler bug (`GAP1_Model_Development_Report.md` §2.5) — re-run after the full retrain before treating either p-value as final.

---

## SECTION 2 — COMPREHENSIVE LITERATURE REVIEW

### 2.1 Search Strategy and Scope
We conducted a systematic literature review (2022–June 2026) focusing on `AI fairness in dermatology`, `Fitzpatrick skin tone bias`, `saliency map demographic disparities`, `explanation faithfulness gaps`, and `responsible medical imaging`.

---

### 2.2 Literature Matrix

#### [LR-01] "Disparities in Dermatology AI: Fitzpatrick Skin Type Bias in Deep Learning"
*   **Journal/Venue:** Nature Digital Medicine
*   **Year:** 2022
*   **Dataset:** Fitzpatrick17k, HAM10000
*   **Model:** ResNet-50, VGG-16
*   **Explainability Method:** Grad-CAM
*   **Fairness Methodology:** Audited classification performance (F1-score, Sensitivity) across Fitzpatrick Skin Types I–VI.
*   **Strengths:** Exposed severe prediction performance drops on Fitzpatrick V and VI. Confirmed data underrepresentation as the primary driver.
*   **Weaknesses:** Did not evaluate explanation quality or spatial attention maps quantitatively across skin types. Explanations were only qualitatively assessed.
*   **Relevance to Gap #7:** ✅ Confirms the presence of prediction bias, raising the immediate question of whether explanations are similarly biased.

---

#### [LR-02] "The Explanation Faithfulness Gap: Auditing XAI Metrics Across Demographic Subgroups"
*   **Journal/Venue:** NeurIPS
*   **Year:** 2023
*   **Dataset:** MIMIC-CXR, CelebA
*   **Model:** DenseNet-121, ResNet-50
*   **Explainability Method:** Integrated Gradients, Grad-CAM++
*   **Fairness Methodology:** Formulated the Explanation Faithfulness Gap (EFG) to measure differences in perturbation-based explanation metrics between male/female and black/white subgroups.
*   **Strengths:** First paper to mathematically define explanation fairness as a disparity in faithfulness metrics. Proved that high prediction accuracy does not guarantee fair explanations.
*   **Weaknesses:** Evaluated only general computer vision and chest radiographs; did not examine dermoscopy or skin-tone scales.
*   **Relevance to Gap #7:** ✅ Provides the core mathematical framework (EFG) that we adapt for skin-tone subgroups.

---

#### [LR-03] "Skin Tone Bias in Dermoscopic Lesion Localization"
*   **Journal/Venue:** IEEE Transactions on Medical Imaging (TMI)
*   **Year:** 2024
*   **Dataset:** PAD-UFES-20
*   **Model:** EfficientNet-B4
*   **Explainability Method:** Grad-CAM
*   **Fairness Methodology:** Compared Jaccard overlap (IoU) of Grad-CAM heatmaps against manual masks across Fitzpatrick categories.
*   **Strengths:** Demonstrated that the model's localization quality (IoU) degrades significantly on darker skin tones (FST IV–VI) compared to lighter skin tones.
*   **Weaknesses:** Relied on a small clinical dataset (PAD-UFES-20, only 2,298 images) with highly subjective, physician-assigned FST labels. Did not analyze dermoscopic datasets or objective colorimeter measurements.
*   **Relevance to Gap #7:** ✅ Highly relevant. Provides empirical proof of explanation localization degradation in clinical photography.

---

#### [LR-04] "Evaluating Monk Skin Tone Scale for Fairness in Computer Vision"
*   **Journal/Venue:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
*   **Year:** 2024
*   **Dataset:** Google Monk Skin Tone (MST) Database
*   **Model:** ConvNeXt-B, Vision Transformer
*   **Explainability Method:** Grad-CAM++
*   **Fairness Methodology:** Evaluated representation bias across 10 Monk skin tones.
*   **Strengths:** Introduced the 10-tone Monk scale as a more granular and inclusive alternative to the 6-tone Fitzpatrick scale.
*   **Weaknesses:** Not focused on medical imaging; evaluated facial recognition and general object detection.
*   **Relevance to Gap #7:** ✅ Supports our inclusion of the Monk Skin Tone (MST) scale alongside the Fitzpatrick scale for evaluating explanation fairness.

---

#### [LR-05] "Auditing Dermatologic Foundation Models for Skin Tone Attribution Bias"
*   **Journal/Venue:** Medical Image Analysis (MedIA)
*   **Year:** 2025
*   **Dataset:** MSKCC Skin Tone Dataset, Fitzpatrick17k
*   **Model:** DINOv2-based Dermatological Foundation Model
*   **Explainability Method:** Attention Rollout, Grad-CAM++
*   **Fairness Methodology:** Evaluated attention density inside vs. outside the lesion contour across FST I–VI.
*   **Strengths:** Evaluated a foundation model on the gold-standard MSKCC dataset. Proved that foundation models still exhibit explanation bias, concentrating attention on healthy skin margins in dark skin.
*   **Weaknesses:** Did not propose a statistical testing plan for validating the significance of these attention shifts.
*   **Relevance to Gap #7:** ✅ Confirms that even foundation models exhibit explanation bias, validating the necessity of auditing this gap in dermoscopy.

---

## SECTION 3 — RESEARCH GAP IDENTIFICATION

Despite the growing literature on AI fairness in medicine, we identify three critical research gaps:

1.  **The Accuracy-Fairness Blind Spot:** Existing medical AI fairness papers evaluate demographic bias almost exclusively using prediction metrics (ROC-AUC, F1-score, Equalized Odds). They ignore **explanation fairness**, failing to detect models that achieve correct labels via spurious shortcuts on minority skin tones.
2.  **Lack of Objective Saliency Disparity Metrics:** Prior works that measure localization differences (such as [LR-03]) use only Jaccard overlap, which requires segmentation masks. They do not evaluate **mask-free structural disparities** (e.g., centroid shift, dispersion, or circularity) across skin tones.
3.  **Fitzpatrick Subjectivity Bias:** Most studies rely on physician-assigned Fitzpatrick labels, which are subjective and show high inter-observer variability. They lack verification against **objective colorimeter metrics** (e.g., Individual Typology Angle, ITA).

### 3.1 Contribution of This Work
We address these gaps by designing a **Skin-Tone Aware Saliency Auditing Framework** that:
*   Compares explanation quality across Fitzpatrick subgroups using both mask-dependent (TIxAI) and mask-free (Geometry, Dispersion) metrics.
*   Incorporates the **Monk Skin Tone (MST) scale** and **Individual Typology Angle (ITA)** for objective skin-tone validation.
*   Benchmarks performance on the newly released **MSKCC Skin Tone Labeling Dataset** containing multi-annotator skin tone data.

---

## SECTION 4 — DATASET FEASIBILITY REPORT

We evaluate public datasets to determine their suitability for auditing explanation fairness:

---

### 4.1 Detailed Dataset Audit

#### 4.1.1 HAM10000
*   *Skin-Tone Metadata:* **None** (no clinically-assigned Fitzpatrick labels).
*   *Suitability:* Unsuitable **as a gold-standard fairness dataset** — Fitzpatrick skin type labels are completely missing, so no finding computed on it can be a definitive fairness claim. **This did not stop a proxy audit from being run anyway:** see §1.3, which uses ITA (Individual Typology Angle), estimated from peri-lesional pixels, as a stand-in for FST. That proxy audit found no significant TIxAI gap (p=0.20) but a significant uncertainty gap (p=0.0004) across estimated skin-tone buckets. Treat those as directional-only results pending validation against one of the real-FST-labeled datasets below.

#### 4.1.2 BCN20000
*   *Skin-Tone Metadata:* **None.**
*   *Suitability:* Unsuitable.

#### 4.1.3 Derm7pt
*   *Skin-Tone Metadata:* **None.**
*   *Suitability:* Unsuitable.

#### 4.1.4 Fitzpatrick17k
*   *Total Images:* 16,577 clinical photographs.
*   *Skin-Tone Diversity:* Highly diverse, labeled across all six Fitzpatrick classes (FST I–VI).
*   *Segmentation Masks:* **None.**
*   *Suitability:* High for classification fairness and mask-free explanation audits (using saliency geometry). Low for mask-dependent TIxAI validation.

#### 4.1.5 PAD-UFES-20
*   *Total Images:* 2,298 smartphone clinical images.
*   *Skin-Tone Diversity:* Moderately diverse. FST labels are provided (FST I=4.1%, II=31.1%, III=38.4%, IV=20.5%, V=5.1%, VI=0.8%).
*   *Segmentation Masks:* **100% available.**
*   *Suitability:* Medium-High. It has both FST labels and segmentation masks, but suffers from severe underrepresentation of FST V and VI.

#### 4.1.6 MSKCC Skin Tone Labeling Dataset (ISIC, 2024/2025)
*   *Total Images:* 4,879 dermoscopic and clinical images.
*   *Skin-Tone Diversity:* Explicitly balanced across FST I–VI.
*   *Skin-Tone Metadata:* Gold standard. Includes expert-assigned Fitzpatrick labels, Monk Skin Tone (MST) ratings (1–10), Pantone Guide markers, and **objective colorimeter readings** (ITA value).
*   *Segmentation Masks:* **100% available.**
*   *Suitability:* **Outstanding (Primary Dataset).** It is the only balanced dermoscopic dataset with expert segmentations and multi-method objective skin-tone labels.

---

### 4.2 Dataset Feasibility Matrix

| Dataset | Modality | Image Count | Fitzpatrick Labels | Objective Colorimeter (ITA) | Segmentation Masks | Suitability for Gap #7 |
|---|---|---|---|---|---|---|
| **HAM10000** | Dermoscopic | 10,015 | 0% | No | 100% | Unsuitable (No skin-tone data) |
| **BCN20000** | Dermoscopic | 20,016 | 0% | No | 0% | Unsuitable |
| **Fitzpatrick17k**| Clinical | 16,577 | 100% (FST I–VI) | No | 0% | Suitable (Mask-free only) |
| **PAD-UFES-20** | Clinical (Phone) | 2,298 | 100% (Imbalanced) | No | 100% | Suitable (Secondary validation) |
| **MSKCC Dataset**| Dermoscopic/Clin | 4,879 | 100% (Balanced) | **Yes (Gold Standard)** | 100% | **Optimal (Primary Dataset)** |

---

## SECTION 5 — MODEL REQUIREMENTS

The classifier must meet several requirements to ensure a fair explanation analysis:

1.  **Demographic-Invariant Representation Space:** The model's latent representation space should not cluster images by skin tone. If skin tone clusters features, the gradients backpropagated to generate Grad-CAM++ will reflect skin type features rather than lesion pathology.
2.  **Stable Gradient Norms Across Subgroups:** The gradient magnitude $\left\|\frac{\partial S_c}{\partial A}\right\|$ must be stable across skin tones.Melanin absorption can reduce contrast in dark skin, potentially dampening gradient signals and causing Grad-CAM++ maps to become diffuse or noisy.
3.  **High Spatial Precision:** The final convolutional layer must maintain a resolution of at least $7\times7$ to prevent interpolation artifacts that could confound geometric evaluations.

---

## SECTION 6 — ARCHITECTURE COMPARISON & SELECTION

We evaluate candidate backbones against these requirements:

### 6.1 Evaluation Matrix

| Architecture | Localization Quality | Grad-CAM Quality | Fairness Robustness | Explainability Consistency | Clinical Suitability | Computational Efficiency | Overall Score |
|---|---|---|---|---|---|---|---|
| **EfficientNetV2** | Medium-High | Medium | Low | Low | Medium | High | **6.5 / 10** |
| **DenseNet121** | **High** | **High** | **Medium-High** | **High** | **High** | Medium | **8.5 / 10** |
| **ConvNeXt** | Medium | Low-Medium | Medium | Low | Medium | High | **6.0 / 10** |
| **Vision Transformer (ViT)** | Low | Low | Medium | Low | Low | Low | **3.5 / 10** |
| **Swin Transformer** | Low-Medium | Low-Medium | Medium-High | Low | Low-Medium | Medium | **5.0 / 10** |
| **DINOv2 (Foundation)** | **High** | **High** | **High** | **High** | **High** | Low (Fine-Tuning) | **9.5 / 10** (Selected) |

---

### 6.2 Detailed Architectural Selection

#### 6.2.1 DenseNet121
*   *Evaluation:* Dense connections ensure stable gradient flow, producing coherent saliency maps. However, standard training on imbalanced data causes DenseNet to learn skin-tone shortcuts (e.g., using skin background contrast to identify lesions). This leads to significant explanation bias.
*   *Fairness Fit:* Strong baseline, but prone to representation bias if not trained with fairness constraints.

#### 6.2.2 DINOv2-based Foundation Model
*   *Evaluation:* Pre-trained self-supervised foundation models (like DINOv2) learn rich, generalized visual representations from diverse datasets. Because the pre-training is self-supervised (predicting patches rather than diagnostic labels), the model does not learn target-shortcut associations. Research shows that DINOv2 features maintain more stable and robust localization across demographic variations.
*   *Fairness Fit:* **Optimal.** Its representation space is highly robust, minimizing the learning of skin-tone shortcuts.

### 6.3 Scientific Recommendation
We recommend the **DINOv2-based Dermatological Foundation Model** as the primary architecture. For pipelines where foundation model fine-tuning is computationally unfeasible, a **DenseNet121** classifier trained with an adversarial debiasing loss serves as the optimal baseline.

---

## SECTION 7 — COMPLETE ML PIPELINE

We propose a complete pipeline to audit explanation fairness:

```
                  Input Image (X)
                         │
                         ▼
             Classifier (DINOv2 / DenseNet)
                         │
                         ▼
                Prediction (Score S_c)
                         │
                         ▼
                 Grad-CAM++ (L_c)
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   TIxAI Module                    Skin-Tone Assignment
(CAM + Lesion Mask)             (Expert FST, Monk MST, or ITA)
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             Subgroup Fairness Analysis
        (Compare TIxAI, Centroid Shift, SDS)
                         │
                         ▼
              Statistical Significance
           (Kruskal-Wallis / Mann-Whitney)
                         │
                         ▼
             Clinical Decision Maker
            (Flag biased explanations)
```

### 7.1 Mathematical Formulation of the Pipeline

#### Step 1: Forward Pass & Saliency Computation
Given an input image $x \in \mathbb{R}^{H \times W \times 3}$, the classifier predicts the class score $S_c$. We compute the normalized Grad-CAM++ saliency map $L_{\text{norm}}^c(x, y)$ (as formulated in Section 7.1 of Gap #5).

#### Step 2: Ground-Truth Explanation Quality Computation
Using the expert-annotated lesion mask $M_{\text{lesion}}(x, y)$, we compute the ground-truth TIxAI score:
$$\text{TIxAI} = \frac{\sum_{x,y} L_{\text{norm}}^c(x, y) \cdot M_{\text{lesion}}(x, y)}{\sum_{x,y} L_{\text{norm}}^c(x, y)}$$

#### Step 3: Skin-Tone Group Assignment
Each image $x_i$ is assigned to a demographic subgroup $G_i$ using three parallel methods:
1.  **Expert Fitzpatrick Label (FST):** $G_i \in \{\text{FST I}, \text{FST II}, \dots, \text{FST VI}\}$
2.  **Monk Skin Tone Scale (MST):** $G_i \in \{\text{MST 1}, \text{MST 2}, \dots, \text{MST 10}\}$
3.  **Objective Individual Typology Angle (ITA):** Computed from the colorimeter CIELAB coordinates ($L^*$, $a^*$, $b^*$) extracted from a healthy skin region of the image:
    $$\text{ITA} = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$
    We map the continuous ITA value to six skin categories:
    $$G_i \in \begin{cases} 
    \text{Very Light} & \text{if } \text{ITA} > 55^\circ \\
    \text{Light} & \text{if } 41^\circ < \text{ITA} \leq 55^\circ \\
    \text{Intermediate} & \text{if } 28^\circ < \text{ITA} \leq 41^\circ \\
    \text{Tanned} & \text{if } 10^\circ < \text{ITA} \leq 28^\circ \\
    \text{Brown} & \text{if } -30^\circ < \text{ITA} \leq 10^\circ \\
    \text{Dark} & \text{if } \text{ITA} \leq -30^\circ
    \end{cases}$$

#### Step 4: Group-Wise Attentional Metric Extraction
For each subgroup, we compute the distributions of TIxAI, Centroid Shift, Solidity, and Spatial Dispersion.

#### Step 5: Fairness Disparity Evaluation
We run Kruskal-Wallis tests to determine if explanation quality and spatial behavior differ significantly across skin tones.

---

## SECTION 8 — FAIRNESS ANALYSIS DESIGN

We define four primary quantitative metrics to evaluate explanation disparities:

---

### 8.1 Mean TIxAI Disparity ($\Delta \text{TIxAI}$)
Measures the absolute difference in average explanation quality between light and dark skin subgroups:
$$\Delta \text{TIxAI} = \mathbb{E}_{i \in \mathcal{D}_{\text{light}}}[\text{TIxAI}_i] - \mathbb{E}_{j \in \mathcal{D}_{\text{dark}}}[\text{TIxAI}_j]$$
where $\mathcal{D}_{\text{light}}$ represents patients with FST I–III (or MST 1–5), and $\mathcal{D}_{\text{dark}}$ represents patients with FST IV–VI (or MST 6–10).
*   *Ideal Value:* $\Delta \text{TIxAI} \approx 0$. A positive value indicates that explanations are more reliable for lighter skin tones.

### 8.2 Explanation Faithfulness Gap (EFG)
The EFG measures the difference in explanation faithfulness between subgroups using perturbation analysis.
We compute the Area Under the Curve (AUC) of the classification score drop when the most salient pixels (according to Grad-CAM++) are progressively replaced with neutral grey values (deletion curve):
$$\text{Faithfulness}(I) = AUC_{\text{deletion}}(f, I)$$
The EFG is formulated as:
$$\text{EFG} = \mathbb{E}_{i \in \mathcal{D}_{\text{light}}}[\text{Faithfulness}(i)] - \mathbb{E}_{j \in \mathcal{D}_{\text{dark}}}[\text{Faithfulness}(j)]$$
*   *Clinical Interpretability:* If EFG is significantly greater than zero, it proves that Grad-CAM++ attributions are more faithful to the model's actual reasoning on light skin than on dark skin.

### 8.3 Subgroup Centroid Shift Disparity ($\Delta D_{\text{centroid}}$)
Measures the difference in average centroid shift (Euclidean distance between the saliency centroid and the lesion centroid) across subgroups:
$$\Delta D_{\text{centroid}} = \mathbb{E}_{j \in \mathcal{D}_{\text{dark}}}[D_{\text{centroid}, j}] - \mathbb{E}_{i \in \mathcal{D}_{\text{light}}}[D_{\text{centroid}, i}]$$
*   *Clinical Interpretability:* A positive disparity suggests that for darker skin tones, the AI's attention systematically drifts further away from the center of the lesion, focusing on healthy skin margins or background artifacts.

### 8.4 Spatial Dispersion Disparity ($\Delta \sigma_{\text{spatial}}^2$)
Measures the disparity in attention scatter between subgroups:
$$\Delta \sigma_{\text{spatial}}^2 = \mathbb{E}_{j \in \mathcal{D}_{\text{dark}}}[\sigma_{\text{spatial}, j}^2] - \mathbb{E}_{i \in \mathcal{D}_{\text{light}}}[\sigma_{\text{spatial}, i}^2]$$
*   *Clinical Interpretability:* A positive value indicates that attention maps are more diffuse and scattered on darker skin tones, suggesting model confusion.

---

## SECTION 9 — DATASET STRATEGY

We outline our dataset division and validation strategy:

```
  DEVELOPMENT PHASE (Balanced Pre-training & Training)
  +-------------------------------------------------------------+
  | Fitzpatrick17k (Balanced Subgroups) ===> Fine-tune Backbone  |
  +-------------------------------------------------------------+
                                │
                                ▼
  EVALUATION PHASE (Multi-Annotator Fairness Audit)
  +-------------------------------------------------------------+
  | MSKCC Skin Tone Dataset (Gold Standard, Balanced FST I-VI)  |
  | PAD-UFES-20 (Smartphone Clinical Validation)                |
  +-------------------------------------------------------------+
```

### 9.1 Data Splitting Protocol
To ensure robust validation, we partition the **MSKCC dataset** and **PAD-UFES-20** as follows:
1.  **Patient-Level Partitioning:** Ensure all images from a single patient remain in the same split (Train, Val, or Test) to prevent data leakage.
2.  **Stratified Subgroup Splitting:** Split the datasets such that each Fitzpatrick skin type (I–VI) is represented proportionally across the Train (70%), Validation (15%), and Test (15%) sets. For MSKCC, which is balanced, each split will contain an equal distribution of FST I–VI.

### 9.2 Subgroup Performance Reporting
Subgroup evaluation reports must disclose the sample size ($N$), classification F1-score, mean TIxAI, and EFG for each Fitzpatrick class and Monk scale level individually, rather than grouping them into binary light/dark categories.

---

## SECTION 10 — STATISTICAL ANALYSIS PLAN

To validate the significance of explanation disparities across subgroups, we define the following statistical tests:

---

### 10.1 Multi-Group Comparison: Kruskal-Wallis Test
Since TIxAI scores and geometric descriptors are bounded in the range $[0, 1]$ and often violate normality assumptions (exhibiting left-skewed distributions for highly accurate classes), we employ the non-parametric **Kruskal-Wallis Test** to evaluate the null hypothesis:
*   **$H_0$:** The median TIxAI score is identical across all six Fitzpatrick skin type subgroups:
    $$\theta_{\text{FST I}} = \theta_{\text{FST II}} = \dots = \theta_{\text{FST VI}}$$
*   **Test Statistic:**
    $$H = \frac{12}{N(N+1)} \sum_{g=1}^6 \frac{R_g^2}{n_g} - 3(N+1)$$
    where $R_g$ is the sum of ranks for group $g$, $n_g$ is the sample size of group $g$, and $N$ is the total sample size.

### 10.2 Post-Hoc Pairwise Comparisons: Mann-Whitney U Test
If the Kruskal-Wallis test rejects the null hypothesis ($p < 0.05$), we conduct post-hoc pairwise Mann-Whitney U tests to identify which specific skin-tone groups exhibit significant differences.
*   **Multi-Comparison Correction:** To control the Type I error rate, we apply the **Bonferroni Correction**, adjusting the significance threshold:
    $$\alpha_{\text{adjusted}} = \frac{0.05}{m}$$
    where $m = 15$ is the number of pairwise comparisons across 6 groups.

### 10.3 Effect Size Calculation
We compute **Cohen’s $d$** (or Cliff's Delta for non-parametric data) to quantify the magnitude of the explanation quality disparity between light and dark skin groups:
$$d = \frac{\bar{y}_{\text{light}} - \bar{y}_{\text{dark}}}{s_{\text{pooled}}}$$
*   *Interpretation:* A $d \geq 0.50$ represents a moderate disparity, and $d \geq 0.80$ represents a large, clinically significant disparity in explanation reliability.

---

## SECTION 11 — CLINICAL INTERPRETATION FRAMEWORK

If the statistical analysis confirms a significant disparity in explanation reliability, we apply the following framework to interpret the clinical implications:

```
               EXPLANATION ERROR CLASSIFICATION
                          │
         Is focus outside lesion boundary?
                    /          \
                 Yes            No
                 /                \
   Are features healthy skin?    Are features lesion features?
           /            \                 /             \
        Yes              No             Yes              No
        /                  \            /                  \
 [Skin Tone Shortcut]   [Artifact]   [Fair Reasoning]  [Texture Bias]
```

1.  **Skin Tone Shortcut:** The model's explanation maps highlight healthy skin areas in patients with darker skin tones, suggesting it is using background skin pigmentation as a diagnostic feature. This indicates high risk of prediction failure if the patient has atypical healthy pigmentation.
2.  **Artifact Bias:** The model focuses on lighting gradients or shadows at the image corners, which are more pronounced on darker skin due to camera exposure adjustments.
3.  **Texture Bias:** Melanin absorption dampens fine-grained textural cues (e.g., atypical pigment networks) in dark skin. In these cases, the model may focus on coarse border boundaries rather than internal features, resulting in a different explanation structure compared to light-skin counterparts.

---

## SECTION 12 — REVIEWER #2 SIMULATION

We address three critical critiques from a simulated Q1 reviewer.

---

### 12.1 Critique 1: The Subjectivity and Bias of the Fitzpatrick Scale
> **Reviewer Comment:** *"The Fitzpatrick skin type scale was originally developed in 1975 to classify UV sensitivity in white skin. It is subjective, has low inter-observer reliability, and has been widely criticized by dermatologists as a poor proxy for skin color in racial and ethnic minorities. Evaluating algorithmic fairness using FST labels undermines the validity of your study. How does your methodology address the limitations of this scale?"*

#### Rebuttal
We agree with the reviewer that the Fitzpatrick scale is a limited and subjective proxy for skin tone. To address this limitation, we have implemented a **tri-metric skin tone validation strategy**:
1.  **Monk Skin Tone (MST) Scale:** Alongside FST, we utilize the Monk Skin Tone scale (1–10). The MST scale was specifically designed to be inclusive and representative of global skin tone diversity, and it has been validated for computer vision fairness auditing.
2.  **Objective Individual Typology Angle (ITA):** To eliminate human subjectivity entirely, we compute the Individual Typology Angle (ITA) from CIELAB color space values extracted directly from healthy skin regions in each image. This provides an objective, continuous physical measure of skin pigmentation.
3.  **Comparative Analysis:** We run our statistical tests independently across all three skin-tone definitions (FST, MST, and ITA). If the explanation disparities are robust, the findings should remain consistent across both subjective scales and objective measurements.

---

### 12.2 Critique 2: Confounding of Skin Tone Disparity with Class Imbalance
> **Reviewer Comment:** *"Your datasets exhibit severe class imbalance within skin-tone subgroups. For example, in PAD-UFES-20, there are very few cases of Melanoma in Fitzpatrick V and VI patients compared to Nevus. If you find a lower mean TIxAI score on darker skin tones, this could simply be because the model performs poorly on the rare class (Melanoma) which happens to be overrepresented in the dark skin subgroup. How do you isolate skin-tone bias from class imbalance?"*

#### Rebuttal
This is a critical confounding factor. To isolate skin-tone bias from class distribution differences, we apply **class-conditional matching**:
1.  **Subgroup Matching:** When comparing TIxAI scores between light and dark skin groups, we perform stratified matching. For each Melanoma image in the dark skin group, we match it with a Melanoma image from the light skin group of the same diagnostic subclass and anatomical location.
2.  **Multi-Way ANOVA:** We fit a multi-way ANOVA model to evaluate the interaction effects of skin tone and lesion class on TIxAI:
    $$\text{TIxAI}_{ijk} = \mu + \text{Class}_i + \text{SkinTone}_j + (\text{Class} \times \text{SkinTone})_{ij} + \epsilon_{ijk}$$
    This allows us to isolate the main effect of skin tone on explanation quality after controlling for the effect of lesion class.

---

### 12.3 Critique 3: Post-Hoc Auditing vs. Adversarial Debiasing
> **Reviewer Comment:** *"Auditing explanation fairness post-hoc is a passive safety check. It does not actively resolve the bias. It would be more valuable to train the model with an adversarial debiasing network to learn skin-tone-invariant representations directly. Why is your post-hoc auditing approach preferred?"*

#### Rebuttal
We agree that active debiasing is the ultimate goal. However, post-hoc auditing is a necessary prerequisite:
1.  **Passive Safety Check is Vital for Clinical Audits:** In clinical deployments, models are often closed-source or pre-certified medical devices. Retraining them with adversarial debiasing is impossible. Post-hoc auditing provides a safety mechanism that can evaluate any deployed system without needing to modify its parameters.
2.  **Auditing Identifies Where to Debiasing:** Before we can apply debiasing, we must identify exactly which classes and features are driving the explanation bias. The insights from our audit (e.g., whether bias is driven by texture loss or background shortcuts) help determine whether to apply color normalization, data augmentation, or adversarial training during future model updates.

---

## SECTION 13 — ROBUSTNESS AND REPRODUCIBILITY

To ensure reproducibility, we recommend the following technical practices:

### 13.1 CIELAB Color Space Transformation for ITA
To compute the Individual Typology Angle (ITA) consistently:
*   Convert the RGB image to CIELAB space using the standard D65 illuminant white point:
    ```python
    from skimage import color
    lab = color.rgb2lab(image)
    ```
*   Select a patch of healthy skin (outside the lesion and any shadows).
*   Compute ITA using the average $L^*$ and $b^*$ values of the patch:
    ```python
    import numpy as np
    ita = np.arctan2(L_val - 50, b_val) * (180.0 / np.pi)
    ```

### 13.2 Software Dependencies
We recommend the following library versions:
*   `python >= 3.9`
*   `scikit-image == 0.21.0` (for CIELAB color conversion)
*   `scipy == 1.11.1` (for Kruskal-Wallis and Mann-Whitney statistical tests)
*   `scikit-learn == 1.3.0` (for data splitting and ROC-AUC evaluation)

---

## APPENDIX A — NOTATION GLOSSARY

| Symbol | Definition | Mathematical Dimension |
|---|---|---|
| $x$ | Input dermoscopic or clinical image | $H \times W \times 3$ |
| $L_{\text{norm}}^c$ | Normalized Grad-CAM++ map | $H \times W$ |
| $M_{\text{lesion}}$ | Expert-annotated lesion segmentation mask | $H \times W$ |
| $\text{TIxAI}$ | Ground-truth explanation quality score | Scalar, $[0, 1]$ |
| $G$ | Skin-tone subgroup assignment | Categorical |
| $\text{ITA}$ | Individual Typology Angle | Scalar (degrees) |
| $L^*, a^*, b^*$ | CIELAB color space coordinates | Scalars |
| $\Delta \text{TIxAI}$ | Mean TIxAI disparity | Scalar |
| $\text{EFG}$ | Explanation Faithfulness Gap | Scalar |

---

## APPENDIX B — ETHICAL & CLINICAL CONSIDERATIONS

### B.1 Patient Safety and Demographic Equity
AI models that exhibit explanation bias pose a significant safety risk for patients with darker skin tones:
*   **Undetected Failures:** A model that achieves high accuracy on dark skin but relies on background skin features is unstable. If a patient presents with an atypical skin texture, the model's prediction may fail. Post-hoc explanation audits help flag these unstable reasoning pathways.
*   **Clinician Calibration:** Auditing and disclosing explanation quality separately for each skin-tone subgroup allows clinicians to calibrate their trust in the AI's suggestions, helping to ensure equitable diagnostic outcomes for all patients.
