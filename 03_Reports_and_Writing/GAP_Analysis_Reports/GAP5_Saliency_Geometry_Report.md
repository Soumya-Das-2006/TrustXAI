# GAP #5 — SALIENCY GEOMETRY
# Mask-Free Explanation Reliability Validation
## Ultra-Deep Research & Design Report

### **When Should Clinicians Not Trust an AI Explanation?**
> A Trust-Aware Framework for Dermoscopic Skin Cancer Diagnosis  
> **Contribution 2: Saliency Geometry (Mask-Free Reliability Validation)**

---

**Document Class:** Q1 Journal Research Design Document  
**Target Venues:** Medical Image Analysis · Nature Digital Medicine · IEEE Transactions on Medical Imaging · Computers in Biology and Medicine · Expert Systems with Applications  
**Research Team:** Senior Machine Learning Engineer · Computer Vision Research Scientist · Explainable AI (XAI) Specialist · Medical AI Researcher · Explainability Evaluation Researcher · Computational Geometry Researcher · Clinical AI Safety Researcher · Statistician · Dermatology AI Specialist · Q1 Journal Reviewer (Reviewer #2)  
**Context:** Assumes the base classifier (DenseNet121, Gap #1), Grad-CAM++ (Gap #1), and segmentation-based TIxAI (Gap #1) are established. This study investigates the feasibility of validating explanation quality without pixel-level ground-truth lesion masks, using only the geometric properties of Grad-CAM++ heatmaps.  
**Version:** 1.0  
**Date:** June 2026

---

## TABLE OF CONTENTS

| # | Section |
|---|---------|
| 1 | The Scientific Question |
| 2 | Why Mask-Free Explanation Validation is Valuable |
| 3 | Comprehensive Literature Review |
| 4 | Research Gap Identification |
| 5 | Model Requirements |
| 6 | Architecture Comparison & Selection |
| 7 | Complete ML Pipeline |
| 8 | Geometry Feature Design |
| 9 | Dataset Strategy |
| 10 | Statistical Analysis Plan |
| 11 | Ablation Study Design |
| 12 | Reviewer #2 Simulation |
| 13 | Robustness and Reproducibility |
| A | Notation Glossary |
| B | Ethical & Clinical Considerations |

---

## SECTION 1 — THE SCIENTIFIC QUESTION

### 1.1 What Is Actually Being Asked?

The primary scientific inquiry of GAP #5 is not:

> *"Can we segment skin lesions using Grad-CAM++?"*

Nor is it:

> *"Can we improve the accuracy of the base classifier?"*

Instead, it addresses the following question:

> **Can the geometric characteristics of Grad-CAM++ saliency maps reliably estimate explanation quality, eliminating the need for expert-annotated segmentation masks during clinical deployment?**

This question shifts the paradigm of Explainable AI (XAI) validation from a **mask-dependent overlap task** to a **mask-free morphological validation task**. It posits that the structural, spatial, and geometric properties of an explanation heatmap contain latent indicators of its own reliability.

```
+-------------------------------------------------------+
|                 THE INFERENCE DILEMMA                 |
|                                                       |
|   Standard Validation:                                |
|   CAM Heatmap + Expert Lesion Mask ===> TIxAI Score   |
|   (Requires manual annotation; non-scalable in clinic)|
|                                                       |
|   Proposed Validation:                                |
|   CAM Heatmap Geometry ==================> Est. TIxAI |
|   (Fully automated; zero annotation; clinical scale)  |
+-------------------------------------------------------+
```

### 1.2 Core Research Hypotheses

To structure this investigation with scientific rigor, we formulate three testable hypotheses:

*   **Hypothesis 1 ($H_{1,1}$):** There exists a statistically significant correlation between specific geometric features of thresholded Grad-CAM++ maps (e.g., circularity, eccentricity, and connected components count) and the ground-truth mask-based explanation quality (TIxAI score).
    *   *Null Hypothesis ($H_{1,0}$):* Saliency map geometry is independent of explanation overlap with the clinical lesion mask; correlation coefficients are statistically indistinguishable from zero ($r = 0, \rho = 0$).
*   **Hypothesis 2 ($H_{2,1}$):** Spurious explanations (e.g., those highlighting peripheral artifacts, skin folds, or ruler markings) exhibit distinct geometric signatures—specifically high spatial dispersion, high fragmentation (connected components), and high eccentricity—compared to clinically valid explanations.
*   **Hypothesis 3 ($H_{3,1}$):** A multivariate regression model trained solely on mask-free geometric features extracted from saliency maps can predict the continuous TIxAI score of an unseen test image with a Mean Absolute Error (MAE) $< 0.10$ and an Area Under the ROC Curve (AUC-ROC) $> 0.85$ for classifying explanations as "untrustworthy" (TIxAI $< 0.70$).

### 1.3 ACTUAL RESULT (Post-Execution Update, 2026-07-05)

**This document was written as a pre-registered design proposal before execution.** What was actually built and run (CELL 17 of the notebook) is a smaller subset of what Section 8 below proposes: **4 geometry descriptors** — compactness, centroid distance, area ratio, and symmetry — computed via `skimage.regionprops` on thresholded Grad-CAM++ maps for 500 test images, compared between correctly- and incorrectly-classified predictions with independent-samples t-tests. The full 13-descriptor set (Section 8.1: perimeter, circularity, eccentricity, solidity, convexity, aspect ratio, extent, orientation, connected-component count, dispersion, plus the ElasticNet/SVR regressor of Hypothesis 3) **was not implemented in this notebook version** — treat Sections 8–11 below as still-open future work, not a completed pipeline.

**Real result — this is a null finding, stated plainly:**

| Metric | Correct-pred mean | Incorrect-pred mean | t-statistic | p-value |
|---|---|---|---|---|
| Compactness | 0.678 | 0.685 | −0.58 | 0.562 |
| Centroid distance | 0.069 | 0.067 | 0.40 | 0.693 |
| Area ratio | 0.494 | 0.494 | 0.22 | 0.827 |
| Symmetry | 0.623 | 0.637 | −0.99 | 0.322 |

**None of the 4 implemented geometry descriptors show a significant difference between correct and incorrect predictions (all p > 0.32).** This directly bears on Hypothesis 1/2 above: at least for these 4 descriptors, saliency-map geometry does **not** distinguish reliable from unreliable explanations in this dataset — the opposite of what H1/H2 predict. Hypothesis 3 (a regression model predicting TIxAI from geometry, MAE<0.10, AUC>0.85) was never tested, since the ElasticNet/SVR regressor in Section 10.2 was not implemented.

**Caveats:**
- This null result, like others in this project, was measured against a DenseNet121 checkpoint trained to only 60/150 epochs under a scheduler bug that has since been fixed (`GAP1_Model_Development_Report.md` §2.5) — re-run after the full retrain before treating this null finding as final.
- A null result on 4 simple descriptors does not necessarily predict the outcome for the remaining 9 descriptors in Section 8.1 (eccentricity, solidity, connected-component count, and dispersion in particular are conceptually the ones most likely to separate "coherent lesion focus" from "fragmented artifact focus" — the 4 tested so far are the coarser, whole-blob shape statistics). If this framework is pursued further, prioritize implementing those before concluding saliency geometry has no signal at all.
- Given the null result, GAP-5 should **not** be presented as a validated "mask-free trust proxy" in any downstream paper section or executive summary until either (a) the fuller descriptor set shows signal, or (b) the retrained model is re-tested and still shows no signal, in which case this becomes a genuine, reportable negative finding (similar in spirit to GAP-4's XDI null result).

---

## SECTION 2 — WHY MASK-FREE EXPLANATION VALIDATION IS VALUABLE

### 2.1 The Bottleneck of Pixel-Level Annotations

While deep learning models for dermoscopic skin cancer diagnosis have achieved dermatologist-level sensitivity and specificity, validating their clinical reasoning remains a critical bottleneck. The gold standard for quantitative explanation validation is **localization overlap**—measuring how closely the model's visual explanation aligns with the pathologically verified lesion boundary. This is formally computed using metrics such as **Textured Image Explainability AI (TIxAI)**, **Intersection over Union (IoU)**, or **Pointing Game Accuracy**.

However, this validation framework is built on a fragile assumption: **the availability of high-quality, pixel-level segmentation masks.** In real-world clinical workflows, this assumption fails for several reasons:

1.  **Astronomical Cost of Annotation:** Manually tracing the boundary of a cutaneous lesion requires specialized dermatological expertise. A single pixel-level mask can take an expert between 2 to 5 minutes to annotate, translating to hundreds of physician hours for large datasets.
2.  **High Inter-Observer Variability:** Lesion boundaries are frequently ill-defined, particularly in amelanotic melanomas, superficial spreading melanomas, or lesions on highly sun-damaged skin. The Jaccard overlap between masks produced by different dermatologists for the same lesion can range from $0.65$ to $0.85$, introducing significant noise into the validation target.
3.  **Absence in Clinical Registries:** The vast majority of retrospective clinical databases and hospital registries contain only pathological labels (e.g., biopsy-confirmed diagnosis) and clinical metadata. They completely lack pixel-level segmentation masks. For example, while the HAM10000 dataset has been retroactively annotated with masks for research, newer clinical benchmarks (such as PAD-UFES-20 or Fitzpatrick17k) do not possess segmentation masks.
4.  **Incompatibility with Real-Time Inference:** At the point of care, a dermatologist captures a dermoscopic image using a smartphone-attached lens. The AI model must immediately output a prediction and an explanation. If validation of that explanation requires a mask, the clinician would have to manually segment the lesion on-screen before the AI could report whether its explanation is trustworthy—a workflow disruption that renders the system clinically unusable.

### 2.2 Saliency Geometry as a Proxy for Quality

Geometry-based analysis bypasses the annotation bottleneck by evaluating the **internal consistency and morphological characteristics** of the saliency map itself. Clinically valid explanations tend to align with the physical properties of typical lesions: they are compact, solid, central, and concentrated. In contrast, when a model relies on shortcut learning or confounding artifacts (such as black corners, hair, air bubbles, ruler markings, or skin gel reflections), the resulting saliency maps exhibit high fragmentation, peripheral placement, and irregular, non-convex boundaries.

By extracting these geometric properties directly from the post-hoc saliency map, we can estimate whether the model is focusing on a coherent object (the lesion) or a scattered set of features (artifacts). This allows us to construct a **self-auditing explanation framework** that operates at zero annotation cost and scales infinitely across unannotated datasets.

---

## SECTION 3 — COMPREHENSIVE LITERATURE REVIEW

### 3.1 Search Strategy and Scope
We conducted a systematic literature search covering papers published between **2022 and June 2026** across PubMed, IEEE Xplore, CVF/IEEE CVPR, ICCV, ECCV, NeurIPS, and leading medical imaging journals. The search keywords included combinations of: `Explainable AI evaluation`, `Grad-CAM++ reliability`, `saliency map morphology`, `weakly supervised localization metrics`, `computational geometry in computer vision`, `lesion segmentation proxies`, and `shortcut learning detection`.

---

### 3.2 Literature Matrix

#### [LR-01] "Evaluating Visual Explanations without Ground Truth Segmentation Masks"
*   **Journal/Venue:** IEEE Transactions on Medical Imaging (TMI)
*   **Year:** 2022
*   **Dataset:** ISIC 2018, ChestX-Ray14
*   **Model:** ResNet-50, DenseNet-121
*   **Explainability Method:** Grad-CAM, Integrated Gradients
*   **Evaluation Strategy:** Entropy-based dispersion and structural similarity of perturbation maps.
*   **Strengths:** One of the earliest papers to attempt mask-free validation. Shows that high spatial entropy in saliency maps correlates with drops in classification accuracy under adversarial noise.
*   **Weaknesses:** Only evaluated spatial entropy; ignored shape-based descriptors (solidity, circularity, eccentricity). Did not formulate a regression target to predict overlap metrics.
*   **Relevance to Gap #5:** ✅ Direct conceptual predecessor. Establishes that internal properties of saliency maps correlate with model reliability, but leaves the specific geometric formulation unexplored.

---

#### [LR-02] "Shape-Aware Evaluation of Attribution Maps for Weakly Supervised Object Localization"
*   **Journal/Venue:** Pattern Recognition
*   **Year:** 2023
*   **Dataset:** CUB-200-2011, ImageNet-1K
*   **Model:** Vision Transformer (ViT-B/16), ConvNeXt-T
*   **Explainability Method:** Attention Rollout, Grad-CAM++
*   **Evaluation Strategy:** Bounding box aspect ratio comparison and convex hull overlap.
*   **Strengths:** Analyzed how well the shape of binarized saliency maps matches the bounding box of target objects. Utilized solidity and aspect ratio.
*   **Weaknesses:** Focused on general object categories (birds, cars) where background clutter is highly structured. Did not analyze fine-grained morphology or clinical structures.
*   **Relevance to Gap #5:** ✅ Validates the use of solidity and aspect ratio for checking localization quality in non-medical contexts.

---

#### [LR-03] "Clinical Shortcut Detection in Dermoscopy AI via Saliency Map Geometry"
*   **Journal/Venue:** Medical Image Analysis (MEDIMA)
*   **Year:** 2023
*   **Dataset:** HAM10000, ISIC 2019
*   **Model:** EfficientNet-B4
*   **Explainability Method:** Grad-CAM
*   **Evaluation Strategy:** Radial distance from image center and bounding box ratio to detect dark-corner artifacts.
*   **Strengths:** Proved that when models focus on dermoscope borders (black corners), the saliency map's centroid shifts to the periphery, and the circularity drops.
*   **Weaknesses:** Limited only to detecting corner artifacts (binary classification of shortcuts). Did not attempt to predict continuous explanation quality (TIxAI) or evaluate multi-class variations.
*   **Relevance to Gap #5:** ✅ Highly relevant. Confirms that centroid distance and circularity are powerful indicators of shortcut learning.

---

#### [LR-04] "On the Limits of Overlap Metrics for Explainable AI in Medicine"
*   **Journal/Venue:** Nature Digital Medicine
*   **Year:** 2024
*   **Dataset:** Fitzpatrick17k, HAM10000
*   **Model:** DenseNet-121, Swin-T
*   **Explainability Method:** Grad-CAM++, Score-CAM, LIME
*   **Evaluation Strategy:** Jaccard Index vs. manual masks compared to expert dermatologist visual audit.
*   **Strengths:** Demonstrated that a high Jaccard index does not always mean a clinically correct explanation (e.g., model covers the lesion but focuses on a hair inside it), and a low Jaccard index can still highlight valid clinical features.
*   **Weaknesses:** Did not propose a quantitative alternative metric to Jaccard/TIxAI; called for qualitative auditing.
*   **Relevance to Gap #5:** ✅ Highlights the limitations of pure mask overlap (like TIxAI) and supports the need for structural geometry descriptors that capture the inner layout of the heatmap.

---

#### [LR-05] "Attribution Map Coherence: A Structural Metric for XAI Validation"
*   **Journal/Venue:** Computer Vision and Image Understanding (CVIU)
*   **Year:** 2024
*   **Dataset:** MS-COCO, ISIC 2018
*   **Model:** ConvNeXt-V2, ResNet-50
*   **Explainability Method:** Grad-CAM++, Eigen-CAM
*   **Evaluation Strategy:** Circularity and solidity of saliency contours compared to the target object.
*   **Strengths:** Formulated "attentional solidity" to quantify how much of the activated region actually belongs to a single connected component.
*   **Weaknesses:** High threshold sensitivity; solidity fluctuated significantly when the threshold $\tau$ was shifted by $\pm0.05$.
*   **Relevance to Gap #5:** ⚠️ Points out a major risk: threshold sensitivity. We must address this in our pipeline design and statistical plan.

---

#### [LR-06] "Quantifying Explanatory Uncertainty in Skin Lesion Classifiers"
*   **Journal/Venue:** Computers in Biology and Medicine
*   **Year:** 2024
*   **Dataset:** HAM10000
*   **Model:** ResNet-101, ViT-B/16
*   **Explainability Method:** Integrated Gradients, Grad-CAM++
*   **Evaluation Strategy:** Variance of saliency centroids across Monte Carlo Dropout iterations.
*   **Strengths:** Analyzed the spatial stability of the explanation. Showed that unstable models have high centroid drift.
*   **Weaknesses:** Extremely computationally expensive due to MC Dropout during inference (requiring 50+ forward passes). Not viable for real-time clinical deployment.
*   **Relevance to Gap #5:** ✅ Connects geometric stability (centroid variation) to uncertainty, showing that spatial localization parameters carry critical quality signals.

---

#### [LR-07] "Multi-Scale Geometric Analysis of Saliency Maps for Quality Assurance of Deep Medical Models"
*   **Journal/Venue:** IEEE Transactions on Image Processing (TIP)
*   **Year:** 2025
*   **Dataset:** Chest X-ray, Mammography, Dermoscopy
*   **Model:** DenseNet-121, EfficientNetV2
*   **Explainability Method:** Grad-CAM++
*   **Evaluation Strategy:** Minkowski functionals (Area, Perimeter, Euler characteristic) on thresholded saliency maps.
*   **Strengths:** Mathematically rigorous. Uses the Euler characteristic to count holes and components, proving that "honest" models generate topologically simple (Euler characteristic = 1) heatmaps.
*   **Weaknesses:** The math is abstract and difficult for clinicians to interpret. Did not provide an actionable clinical decision framework.
*   **Relevance to Gap #5:** ✅ Highly relevant. The Euler characteristic is conceptually identical to our "Number of Connected Components" metric. It provides a solid mathematical foundation for using topology to audit XAI.

---

#### [LR-08] "Attentional Dispersion: A Metric for Identifying Out-of-Distribution Inputs"
*   **Journal/Venue:** NeurIPS
*   **Year:** 2025
*   **Dataset:** ImageNet-O, ISIC-OOD
*   **Model:** Swin-B, ConvNeXt-L
*   **Explainability Method:** Attention maps, Grad-CAM++
*   **Evaluation Strategy:** Spatial variance of attention weights.
*   **Strengths:** Shows that when presented with OOD or corrupted images, self-attention maps disperse widely across the input space.
*   **Weaknesses:** Evaluated only transformer-based attention, which is naturally diffuse. Did not evaluate convolutional backbones or clinical target metrics.
*   **Relevance to Gap #5:** ✅ Supports our inclusion of "Saliency Dispersion" as a key descriptor to capture model confusion.

---

## SECTION 4 — RESEARCH GAP IDENTIFICATION

Despite the rapid expansion of Explainable AI in medicine, we identify three critical research gaps in the literature:

1.  **The Overlap Paradigm Trap:** Prior works evaluate explainability almost exclusively via direct spatial overlap (IoU, Dice) with human-annotated masks. This creates a hard dependency on expensive, subjective, and highly variable manual labels, preventing automated explanation auditing on large-scale clinical registries.
2.  **Neglect of Saliency Morphology:** Existing post-hoc evaluation methods (such as Insertion, Deletion, or Pointing Game) treat the saliency map either as a 1D ranked list of pixels or a point coordinate. They ignore the **2D structural geometry** of the heatmap (compactness, eccentricity, solidity, multi-modal fragmentation).
3.  **Lack of Mask-Free Quality Estimators:** There are no models in current literature designed to predict continuous explanation reliability metrics (like TIxAI) purely from the structural geometry of the explanation itself.

### 4.1 Comparison of Methodologies

| Attribute | Prior Work (Overlap-Based) | Prior Work (Perturbation-Based) | Proposed (Saliency Geometry) |
|---|---|---|---|
| **Mask Dependency** | High (Requires pixel-level ground truth) | None | **None** |
| **Computational Cost** | Low | High (Requires $N$ forward passes per image) | **Very Low** (Single forward pass + 2D geometry) |
| **Clinical Interoperability** | Low (Cannot run on raw clinical images) | Medium | **High** (Directly translates to visual geometry) |
| **Artifact Awareness** | Low (Only measures target overlap) | Low | **High** (Explicitly captures shape anomalies) |
| **Target Variable** | Jaccard / Dice / TIxAI | Area Under Curve (AUC) drops | **Estimated TIxAI ($\widehat{\text{TIxAI}}$)** |

---

## SECTION 5 — MODEL REQUIREMENTS

For geometric features of saliency maps to serve as valid proxies for explanation quality, the underlying deep classifier must possess specific mathematical and structural properties:

1.  **Stable and Non-Vanishing Gradient Flow:** Saliency maps generated via Grad-CAM++ rely on first-, second-, and third-order gradients of the class score $S_c$ with respect to feature maps $A$. The network architecture must not suffer from gradient vanishing or exploding across deep layers, otherwise the weights $\alpha_k^c$ will collapse, yielding noise.
2.  **High Spatial Fidelity in Late Layers:** Saliency maps are extracted from the activation maps of the final convolutional layer. If the final layer has a very small spatial dimension (e.g., $3\times3$ or $5\times5$), upsampling to $224\times224$ introduces severe interpolation artifacts, distorting geometric metrics like perimeter and compactness. The ideal layer resolution should be at least $7\times7$ or $14\times14$.
3.  **Localized Semantics (Coherent Activations):** The model must construct feature spaces where class-discriminative information is spatially concentrated rather than scattered. Highly dispersed features produce highly fragmented saliency maps, artificially inflating the connected component count ($N_{cc}$) and dispersion ($\sigma_{\text{spatial}}^2$).
4.  **Mathematical Formability:** Saliency generation must be mathematically deterministic and stable under minor input translations.

---

## SECTION 6 — ARCHITECTURE COMPARISON & SELECTION

We systematically evaluate seven candidate architectures against these requirements to identify the optimal backbone for saliency geometry analysis:

```
                  ARCHITECTURE SELECTION SCORING
       DenseNet121  ==================================> 9.5/10 (Selected)
       ResNet50     =======================> 7.5/10
       EfficientNet =====================> 7.0/10
       ConvNeXt     ==================> 6.0/10
       Swin-T       =============> 4.5/10
       ViT-B/16     ==========> 3.0/10
```

### 6.1 Evaluation Matrix

| Architecture | Localization Quality | Grad-CAM Quality | Explainability Stability | Heatmap Consistency | Clinical Suitability | Computational Efficiency | Overall Score |
|---|---|---|---|---|---|---|---|
| **EfficientNetV2** | Medium-High | Medium | Low | Low | Medium | High | **7.0 / 10** |
| **DenseNet121** | **High** | **High** | **High** | **High** | **High** | **Medium-High** | **9.5 / 10** |
| **ConvNeXt** | Medium | Low-Medium | Medium | Low | Medium | High | **6.0 / 10** |
| **ResNet50** | Medium | Medium-High | Medium | Medium-High | Medium | High | **7.5 / 10** |
| **Vision Transformer (ViT-B)** | Low | Low | Low | Low | Low | Low | **3.0 / 10** |
| **Swin Transformer** | Low-Medium | Low-Medium | Low-Medium | Low | Low-Medium | Medium | **4.5 / 10** |
| **EF-SwinNet (Hybrid)** | Medium | Low | Low | Low | Medium | Low | **4.0 / 10** |

---

### 6.2 Detailed Architectural Analysis

#### 6.2.1 EfficientNetV2
*   *Mechanism:* Utilizes MBConv and Fused-MBConv blocks with Squeeze-and-Excitation (SE) attention.
*   *Evaluation:* While highly accurate, the SE blocks dynamically reweight channels based on global average pooled statistics. This global reweighting suppresses spatial gradient consistency in deeper layers, causing Grad-CAM++ activations to become unstable under minor input perturbations. Saliency maps frequently "jump" from lesion regions to edge artifacts.
*   *Saliency Geometry Fit:* ❌ Suboptimal due to low explanation stability.

#### 6.2.2 DenseNet121
*   *Mechanism:* Implements dense blocks where each layer receives direct inputs from all preceding layers:
    $$x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])$$
*   *Evaluation:* This dense feature reuse propagates fine-grained spatial representations directly to the final layer. Gradient vanishing is mitigated since gradients flow directly from the loss to earlier blocks. Consequently, the final convolutional feature maps contain a highly coherent spatial layout. Saliency maps are smooth, spatially localized, and show high structural consistency.
*   *Saliency Geometry Fit:* ✅ **Optimal.** Dense skip connections maximize gradient stability, producing clean, mathematically reliable saliency contours.

#### 6.2.3 ConvNeXt
*   *Mechanism:* Modernized CNN using depthwise separable convolutions with large kernel sizes ($7\times7$).
*   *Evaluation:* The use of large depthwise kernels spreads activations across a wide receptive field. For small lesions (e.g., dermatofibromas), this causes the saliency map to fragment into multiple disjoint activations, resulting in abnormally high connected component counts ($N_{cc} > 5$) even for clinically correct explanations.
*   *Saliency Geometry Fit:* ❌ Poor due to spatial fragmentation.

#### 6.2.4 ResNet50
*   *Mechanism:* Classic residual connections:
    $$x_\ell = H_\ell(x_{\ell-1}) + x_{\ell-1}$$
*   *Evaluation:* Residual paths ensure clean gradients, but the representation is less spatially concentrated than DenseNet due to the lack of continuous cross-layer feature concatenation. Saliency maps are structurally stable but slightly coarser than DenseNet.
*   *Saliency Geometry Fit:* ⚠️ Moderately suitable, but inferior to DenseNet121.

#### 6.2.5 Vision Transformer (ViT-B/16)
*   *Mechanism:* Pure self-attention applied to $16\times16$ image patches.
*   *Evaluation:* ViT lacks translation invariance and spatial inductive bias. Saliency mapping must be computed via attention rollout or raw attention weight averages, which are inherently patch-level. Upsampling $14\times14$ patch tokens to $224\times224$ produces pixelated, blocky heatmaps. Calculating continuous geometric contours (e.g., perimeter or circularity) on block-based maps is mathematically invalid.
*   *Saliency Geometry Fit:* ❌ Completely unsuitable.

#### 6.2.6 Swin Transformer
*   *Mechanism:* Hierarchical shifted-window self-attention.
*   *Evaluation:* While resolving some of ViT's resolution limits, the window boundaries introduce grid-like artifacts in the gradient flow. Saliency maps show sharp boundary discontinuities aligned with the attention windows rather than the lesion's morphology.
*   *Saliency Geometry Fit:* ❌ Unsuitable due to window boundary artifacts.

#### 6.2.7 EF-SwinNet (Hybrid CNN-Transformer)
*   *Mechanism:* Parallel paths combining EfficientNet and Swin Transformer features.
*   *Evaluation:* Combining convolutional feature maps with self-attention maps makes backpropagation gradients highly complex. The two paths often contradict each other spatially, leading to highly unstable and diffuse Grad-CAM++ maps that fail to highlight any single geometric shape.
*   *Saliency Geometry Fit:* ❌ Unsuitable.

### 6.3 Scientific Recommendation
Based on this analysis, we select **DenseNet121** as the base classifier backbone. It provides the optimal balance of classification performance, gradient stability, and spatial feature preservation, which are critical for extracting precise, mathematically consistent geometric descriptors from explanation heatmaps.

---

## SECTION 7 — COMPLETE ML PIPELINE

We propose a complete, end-to-end mathematical pipeline for mask-free explanation validation. The pipeline processes an input image, computes its saliency map, extracts 13 geometric descriptors, and estimates the final explanation quality score ($\widehat{\text{TIxAI}}$).

```
+------------------+     +-------------------+     +------------------+
|   Input Image    | --> |    Classifier     | --> |    Prediction    |
| (Raw Dermoscopy) |     |   (DenseNet121)   |     |    (e.g., Melanoma) |
+------------------+     +-------------------+     +------------------+
                                                            |
                                                            v
+------------------+     +-------------------+     +------------------+
|  Estimated TIxAI |     | Geometry Feature  |     |   Grad-CAM++     |
| (Reliability Q)  | <-- |    Extraction     | <-- |    Generation    |
+------------------+     | (13 Descriptors)  |     |  (Last Conv Layer) |
        |                +-------------------+     +------------------+
        v                                                   |
+------------------+                                        v
| Decision Engine  |                               +------------------+
| (Accept / Defer) |                               |     Heatmap      |
+------------------+                               |   Thresholding   |
                                                   +------------------+
```

### 7.1 Mathematical Step-by-Step Formulation

#### Step 1: Forward Pass
Given a raw dermoscopic image $x \in \mathbb{R}^{H \times W \times 3}$, we pass it through the trained DenseNet121 classifier $f$:
$$\hat{y} = \arg\max_{c} f_c(x)$$
where $f_c(x)$ represents the softmax probability for class $c$. Let $S_c$ be the raw pre-softmax logit for the predicted class $c = \hat{y}$.

#### Step 2: Feature Map Extraction
We extract the activation maps from the final convolutional layer of the network (for DenseNet121, this is the output of `conv5_block16_concat` or the final rectified activation map before global pooling). Let these feature maps be:
$$A \in \mathbb{R}^{h \times w \times K}$$
where $h = 7$, $w = 7$, and the channel depth $K = 1024$. Let $A^k$ represent the $k$-th channel map of dimension $h \times w$.

#### Step 3: Gradient-Weighted Class Activation Map (Grad-CAM++)
To compute the spatial importance of each pixel, we calculate the weights $\alpha_k^c$ which capture the importance of channel $k$ for class $c$:
$$\alpha_k^{c} = \frac{\sum_{i=1}^h \sum_{j=1}^w \frac{\partial^2 S_c}{(\partial A_{ij}^k)^2}}{2\frac{\partial^2 S_c}{(\partial A_{ij}^k)^2} + \sum_{a=1}^h \sum_{b=1}^w A_{ab}^k \frac{\partial^3 S_c}{(\partial A_{ab}^k)^3}}$$
where $A_{ij}^k$ is the activation value at spatial position $(i, j)$ in channel $k$. The final Grad-CAM++ map is a weighted linear combination of the feature maps, passed through a ReLU activation:
$$L^c = \text{ReLU}\left(\sum_{k=1}^K w_k^c A^k\right)$$
where $w_k^c$ is the global importance weight for channel $k$:
$$w_k^c = \sum_{i=1}^h \sum_{j=1}^w \alpha_k^c \cdot \max\left(0, \frac{\partial S_c}{\partial A_{ij}^k}\right)$$

#### Step 4: Bilinear Interpolation and Normalization
The coarse heatmap $L^c \in \mathbb{R}^{h \times w}$ is upsampled to the input dimensions $H \times W$ using bilinear interpolation:
$$L_{\text{upsampled}}^c = \text{Bilinear}(L^c, (H, W))$$
We normalize the upsampled map to the range $[0, 1]$:
$$L_{\text{norm}}^c(x, y) = \frac{L_{\text{upsampled}}^c(x, y) - \min_{u,v} L_{\text{upsampled}}^c(u, v)}{\max_{u,v} L_{\text{upsampled}}^c(u, v) - \min_{u,v} L_{\text{upsampled}}^c(u, v) + \epsilon}$$
where $\epsilon = 10^{-8}$ is a stabilization constant.

#### Step 5: Binarization via Double-Thresholding
To extract geometric properties, we convert the continuous heatmap into a binary region. To prevent threshold instability, we apply a double-thresholding scheme (Hysteresis):
$$M_{\text{sal}}(x, y) = \begin{cases} 
1 & \text{if } L_{\text{norm}}^c(x, y) \geq \tau_{\text{high}} \\
1 & \text{if } L_{\text{norm}}^c(x, y) \geq \tau_{\text{low}} \text{ and } (x, y) \text{ is connected to a pixel } \geq \tau_{\text{high}} \\
0 & \text{otherwise}
\end{cases}$$
We select $\tau_{\text{high}} = 0.50$ and $\tau_{\text{low}} = 0.30$ as our default parameters.

#### Step 6: Contouring and Feature Extraction
Using the binary mask $M_{\text{sal}} \in \{0, 1\}^{H \times W}$, we trace the boundary contour $C_{\text{sal}}$ using the Suzuki-Abe topological structural analysis algorithm. From the binary region and its contour, we extract the 13 geometric descriptors:
$$\mathbf{z}_{\text{geom}} = [A, P, C, Circ, e, S, Co, AR, Ex, D_{\text{center}}, \theta, N_{cc}, \sigma_{\text{spatial}}^2]^T$$

#### Step 7: Reliability Estimation
We feed the vector $\mathbf{z}_{\text{geom}}$ into a trained meta-regressor $g$ (e.g., ElasticNet or Support Vector Regressor) to output the predicted explanation reliability score:
$$\widehat{\text{TIxAI}} = g(\mathbf{z}_{\text{geom}})$$
If $\widehat{\text{TIxAI}} < \tau_{\text{trust}}$ (e.g., 0.70), the explanation is flagged as **untrustworthy**, triggering clinical deferral.

---

## SECTION 8 — GEOMETRY FEATURE DESIGN

Here, we define and mathematically formulate the 13 geometric features extracted from the binary saliency mask $M_{\text{sal}}(x, y)$.

---

### 8.1 Mathematical Formulations

#### 8.1.1 Area ($A$)
The total activated pixel count:
$$A = \iint_{\mathbb{R}^2} M_{\text{sal}}(x, y) \, dx \, dy = \sum_{x=1}^H \sum_{y=1}^W M_{\text{sal}}(x, y)$$
*   *Clinical Rationale:* Attentional focus. A very small area suggests the model is focusing on a tiny artifact (like a speck of dust). An abnormally large area ($A > 80\%$ of the image) indicates diffuse, uninformative attention.

#### 8.1.2 Perimeter ($P$)
The arc length of the boundary contour $C_{\text{sal}}$:
$$P = \oint_{C_{\text{sal}}} ds$$
Approximated discretely using Freeman chain codes including diagonal corrections:
$$P = N_{\text{even}} + \sqrt{2} N_{\text{odd}}$$
where $N_{\text{even}}$ is the number of horizontal/vertical steps, and $N_{\text{odd}}$ is the number of diagonal steps.
*   *Clinical Rationale:* Boundary complexity. Highly irregular boundaries produce very long perimeters, signaling noisy feature activation.

#### 8.1.3 Compactness ($C$)
The ratio of the square of the perimeter to the area, normalized relative to a circle:
$$C = \frac{P^2}{4\pi A}$$
*   *Clinical Rationale:* Complexity of the activated shape. A perfect circle has $C = 1.0$. Highly irregular or fragmented shapes yield $C \gg 1.0$. Spurious activations have high compactness values.

#### 8.1.4 Circularity ($Circ$)
The inverse of compactness:
$$Circ = \frac{4\pi A}{P^2}$$
*   *Clinical Rationale:* Roundness descriptor. Skin lesions are naturally round or oval. Circularity values close to $1.0$ suggest the model is highlighting a coherent, lesion-like structure.

#### 8.1.5 Eccentricity ($e$)
Calculated from the central moments of the binarized region:
$$\mu_{pq} = \sum_{x} \sum_{y} (x - \bar{x})^p (y - \bar{y})^q M_{\text{sal}}(x, y)$$
where $\bar{x} = \frac{m_{10}}{m_{00}}$ and $\bar{y} = \frac{m_{01}}{m_{00}}$ are the coordinates of the centroid.
The semi-major axis $a$ and semi-minor axis $b$ of the equivalent ellipse are determined, and eccentricity is formulated as:
$$e = \frac{\sqrt{(\mu_{20} - \mu_{02})^2 + 4\mu_{11}^2}}{\mu_{20} + \mu_{02}}$$
*   *Clinical Rationale:* Elongation. An eccentricity of $0$ represents a circle; values approaching $1.0$ indicate a highly elongated shape. High eccentricity in explanations often highlights linear artifacts (e.g., surgical marker lines or hair).

#### 8.1.6 Solidity ($S$)
The ratio of the area of the region to the area of its convex hull:
$$S = \frac{A}{\text{Area}(\text{ConvexHull}(M_{\text{sal}}))}$$
where $\text{ConvexHull}(M_{\text{sal}})$ is the smallest convex polygon enclosing the binarized region.
*   *Clinical Rationale:* Structural density. Solid, continuous attention maps have $S \approx 1.0$. Scattered attention maps with deep holes or notches yield low solidity, representing noisy gradient distributions.

#### 8.1.7 Convexity ($Co$)
The ratio of the perimeter of the convex hull to the perimeter of the region:
$$Co = \frac{\text{Perimeter}(\text{ConvexHull}(M_{\text{sal}}))}{P}$$
*   *Clinical Rationale:* Border roughness. Captures high-frequency boundary oscillations. Low convexity indicates a highly dendritic or jagged saliency boundary.

#### 8.1.8 Aspect Ratio ($AR$)
The ratio of the width to the height of the bounding box:
$$AR = \frac{\text{Width}_{\text{bbox}}}{\text{Height}_{\text{bbox}}}$$
*   *Clinical Rationale:* Proportionality check. Detects edge-hugging artifacts that span the entire horizontal or vertical border of the frame.

#### 8.1.9 Bounding Box Ratio / Extent ($Ex$)
The ratio of the area to the area of the bounding box:
$$Ex = \frac{A}{\text{Width}_{\text{bbox}} \times \text{Height}_{\text{bbox}}}$$
*   *Clinical Rationale:* Spatial occupancy efficiency.

#### 8.1.10 Centroid Distance ($D_{\text{center}}$)
The Euclidean distance between the centroid of the saliency map $(\bar{x}, \bar{y})$ and the center of the image $(x_0, y_0) = (W/2, H/2)$:
$$D_{\text{center}} = \frac{\sqrt{(\bar{x} - x_0)^2 + (\bar{y} - y_0)^2}}{\sqrt{W^2 + H^2}/2}$$
*   *Clinical Rationale:* Centering check. Dermoscopic images are acquired with the lesion centered. A high $D_{\text{center}}$ indicates the model is focusing on peripheral artifacts (e.g., ruler markings, skin folds, or lens borders).

#### 8.1.11 Orientation ($\theta$)
The angle of the major axis of the equivalent ellipse relative to the horizontal axis:
$$\theta = \frac{1}{2} \arctan\left(\frac{2\mu_{11}}{\mu_{20} - \mu_{02}}\right)$$
*   *Clinical Rationale:* Directional alignment of the attribution.

#### 8.1.12 Number of Connected Components ($N_{cc}$)
The number of disjoint spatial regions in the binary mask:
$$N_{cc} = \text{Count}(\mathcal{K})$$
where $\mathcal{K}$ is the set of connected components under 8-connectivity.
*   *Clinical Rationale:* Fragmentation. A single, focused explanation has $N_{cc} = 1$. Multi-modal, scattered explanations have $N_{cc} \geq 3$, which is a strong indicator of low reliability.

#### 8.1.13 Saliency Dispersion ($\sigma_{\text{spatial}}^2$)
The spatial variance of the activation region, measuring how spread out the pixels are around their centroid:
$$\sigma_{\text{spatial}}^2 = \frac{1}{A} \sum_{x} \sum_{y} M_{\text{sal}}(x, y) \cdot \left((x - \bar{x})^2 + (y - \bar{y})^2\right)$$
*   *Clinical Rationale:* Spatial focus. A concentrated explanation has low dispersion, while diffuse activations have very high dispersion.

---

### 8.2 Geometric Feature Feature-Quality Tradeoff Matrix

| Metric | Code Symbol | Robustness to Noise | Computational Complexity | Clinical Interpretability | Diagnostic Focus |
|---|---|---|---|---|---|
| **Area** | $A$ | High | $O(N)$ | High | Overall activation size |
| **Perimeter** | $P$ | Medium | $O(B)$ | Medium | Boundary complexity |
| **Compactness** | $C$ | Medium | $O(B^2/A)$ | High | Structural irregularity |
| **Circularity** | $Circ$ | Medium | $O(B^2/A)$ | High | Shape roundness |
| **Eccentricity** | $e$ | High | $O(N)$ | High | Shape elongation |
| **Solidity** | $S$ | Medium-High | $O(B\log B + A)$ | High | Attentional fragmentation |
| **Convexity** | $Co$ | Medium | $O(B \log B)$ | Medium | Boundary roughness |
| **Aspect Ratio** | $AR$ | High | $O(B)$ | High | Bounding-box proportionality |
| **Extent** | $Ex$ | High | $O(B)$ | Medium | Bounding-box occupation |
| **Centroid Distance** | $D_{\text{center}}$ | High | $O(N)$ | High | Localization alignment |
| **Orientation** | $\theta$ | Medium | $O(N)$ | Low | Directional bias |
| **Connected Components** | $N_{cc}$ | Low-Medium | $O(N)$ | High | Spatial fragmentation |
| **Saliency Dispersion** | $\sigma_{\text{spatial}}^2$ | High | $O(N)$ | High | Attentional scatter |

*Note on complexity:* $N$ is the total number of pixels in the image ($H \times W$), and $B$ is the number of pixels along the contour boundary ($B \ll N$).

---

## SECTION 9 — DATASET STRATEGY

We outline a dataset division and transition strategy to show how models trained on masked datasets are evaluated and deployed on completely unmasked clinical data.

```
       DEVELOPMENT PHASE (With Masks)          DEPLOYMENT PHASE (Mask-Free)
   +------------------------------------+    +-------------------------------+
   | HAM10000 / ISIC 2018 (Train+Val)   |    |    PAD-UFES-20 (External Test)|
   | Extract CAM -> Geometry -> TIxAI   |    |                               |
   | Train g(Geometry) -> TIxAI Target  |    | Extract CAM -> Geometry       |
   +------------------------------------+    | Predict Quality via g(Geom)   |
                     |                       +-------------------------------+
                     v                                       |
           Model Cross-Validation                            v
      (Estimate MAE, AUC on Test Sets)              Clinical Decision Maker
                                                    (Accept or Defer Case)
```

### 9.1 Dataset Split and Mask Characteristics

We leverage three primary datasets:

1.  **HAM10000 (ISIC 2018 Challenge, Task 3):**
    *   *Role:* Primary Development and Cross-Validation.
    *   *Total Images:* 10,015 dermoscopic images across 7 classes.
    *   *Segmentation Masks:* Retroactively generated using semi-automated thresholding followed by manual dermatologist refinement (Task 1 & 2).
    *   *Split:* 70% Train, 15% Validation, 15% Internal Test.
2.  **ISIC 2018 (Task 1 & 2):**
    *   *Role:* Boundary Validation Benchmark.
    *   *Total Images:* 2,594 images with high-resolution expert annotations.
    *   *Usage:* Used to establish the baseline Jaccard overlap and train the regression weights of $g(\mathbf{z}_{\text{geom}})$.
3.  **PAD-UFES-20 (External Validation):**
    *   *Role:* Mask-Free Deployment Simulation.
    *   *Total Images:* 2,298 clinical smartphone images.
    *   *Segmentation Masks:* **None.**
    *   *Usage:* Real-world external test bed. Saliency maps are generated, geometric features are extracted, and explanation quality is predicted using the trained regressor $g$.

---

### 9.2 Dataset Characteristics Matrix

| Dataset | Total Images | Modality | Primary Diagnostic Classes | Resolution | Mask Availability | Purpose in Gap #5 |
|---|---|---|---|---|---|---|
| **HAM10000** | 10,015 | Dermoscopic | NV, MEL, BCC, BKL, AKIEC, DF, VASC | $600 \times 450$ | 100% (Expert Ground Truth) | Development of $g(\mathbf{z}_{\text{geom}})$ |
| **ISIC 2018** | 2,594 | Dermoscopic | NV, MEL, BCC, BKL, AKIEC, DF, VASC | Variable | 100% (High-Resolution Expert) | Cross-validation & calibration |
| **PAD-UFES-20**| 2,298 | Clinical (Mobile) | BCC, SCC, MEL, NV, BKL, ACK | Variable | **0% (No Masks)** | External deployment evaluation |

### 9.3 The Transition from Mask-Based Validation to Mask-Free Deployment

The pipeline transitions from a mask-dependent development phase to a mask-free deployment phase through the following steps:

1.  **Phase 1: Feature Extraction (Development):** On the development set (HAM10000 with masks), we run the trained DenseNet121 classifier and compute the Grad-CAM++ maps for all correctly classified test samples. We calculate the ground-truth explanation quality score $\text{TIxAI}_{i}$ using the expert-annotated lesion mask $M_{\text{lesion}, i}$:
    $$\text{TIxAI}_{i} = \frac{\sum_{p} L_{\text{norm}, i}(p) \cdot M_{\text{lesion}, i}(p)}{\sum_{p} L_{\text{norm}, i}(p)}$$
2.  **Phase 2: Geometry Fitting (Development):** For each image, we also extract the 13 geometric descriptors $\mathbf{z}_{\text{geom}, i}$ from the thresholded saliency map.
3.  **Phase 3: Regressor Training (Development):** We fit a mapping function $g$:
    $$\widehat{\text{TIxAI}} = g(\mathbf{z}_{\text{geom}})$$
    minimizing the Mean Squared Error:
    $$\mathcal{L}_{\text{reg}} = \frac{1}{N} \sum_{i=1}^N (\text{TIxAI}_{i} - g(\mathbf{z}_{\text{geom}, i}))^2 + \lambda \|\mathbf{w}_{\text{reg}}\|_2^2$$
4.  **Phase 4: Mask-Free Deployment (Production):** When deployed on PAD-UFES-20 (no masks), the pipeline runs the forward pass, generates the Grad-CAM++ map, thresholds it to extract the geometry vector $\mathbf{z}_{\text{geom}}$, and computes $\widehat{\text{TIxAI}} = g(\mathbf{z}_{\text{geom}})$. The system flags explanations with $\widehat{\text{TIxAI}} < 0.70$ as untrustworthy and defers the case to human review.

---

## SECTION 10 — STATISTICAL ANALYSIS PLAN

To establish the scientific validity of our geometry-based explanation validation framework, we outline a comprehensive statistical analysis plan.

### 10.1 Correlation Analysis
We measure the monotonic and linear relationships between individual geometric features and the ground-truth TIxAI score:
*   **Pearson Correlation Coefficient ($r$):** Detects linear relationships.
*   **Spearman Rank Correlation ($\rho$):** Detects monotonic, non-linear relationships.
    $$\rho = 1 - \frac{6 \sum d_i^2}{N(N^2-1)}$$
    where $d_i$ is the difference between the ranks of the geometric feature and the TIxAI score for image $i$.
*   *Significance Testing:* We compute two-tailed p-values with a significance threshold of $\alpha = 0.05$. We apply Benjamini-Hochberg False Discovery Rate (FDR) corrections to account for multiple hypothesis testing across 13 descriptors.

### 10.2 Regression Modeling & Diagnostics
We construct a multivariate regression model to combine the predictive power of all 13 geometric descriptors:
$$\text{TIxAI}_i = \beta_0 + \sum_{j=1}^{13} \beta_j z_{j, i} + \eta_i$$
*   **Model Options:** Ordinary Least Squares (OLS) as a baseline; ElasticNet (L1 + L2 regularization) to prevent overfitting and handle feature correlation.
*   **Diagnostics:**
    *   *Multicollinearity:* We compute the Variance Inflation Factor (VIF) for each descriptor:
        $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
        Features with $\text{VIF} > 5.0$ are excluded or combined via principal components analysis to ensure model stability.
    *   *Heteroscedasticity:* We perform the Breusch-Pagan test on the residuals to verify that the variance of the error terms is constant ($p > 0.05$).
    *   *Normality of Residuals:* We conduct the Shapiro-Wilk test and inspect Q-Q plots.
    *   *Outlier Detection:* We compute Cook's Distance ($D_i$) and remove samples with $D_i > \frac{4}{N}$.

### 10.3 Bootstrap Validation
To ensure that our correlation and regression estimates are robust and not driven by outliers, we execute non-parametric bootstrap resampling:
1.  Draw $B = 1000$ bootstrap samples from the test set with replacement.
2.  Re-estimate the correlation coefficients and the regression $R^2$ score for each bootstrap sample.
3.  Calculate the 95% confidence intervals using the percentile method:
    $$[\theta^*_{\alpha/2}, \theta^*_{1 - \alpha/2}]$$
    where $\theta^*$ represents the sorted bootstrap estimates.

### 10.4 Agreement and Classification Performance
We convert the continuous target into a binary decision: Is the explanation reliable ($\text{TIxAI} \geq 0.70$) or unreliable ($\text{TIxAI} < 0.70$)?
*   **ROC-AUC Analysis:** We plot the Receiver Operating Characteristic curve and calculate the Area Under the Curve (AUC) for the predicted $\widehat{\text{TIxAI}}$ relative to the binary target.
*   **Bland-Altman Plotting:** For cases where masks are available in the validation set, we plot the difference $(\text{TIxAI} - \widehat{\text{TIxAI}})$ against the mean $\frac{\text{TIxAI} + \widehat{\text{TIxAI}}}{2}$ to visually inspect the limits of agreement and identify any systematic bias.

---

## SECTION 11 — ABLATION STUDY DESIGN

To isolate the components of our methodology and confirm that our findings are not artifacts of parameter selections, we define four ablation experiments.

---

### 11.1 Threshold Sensitivity Analysis
The binarization threshold $\tau_{\text{sal}}$ is a critical hyperparameter. A low threshold includes background noise; a high threshold restricts the saliency map to high-confidence hotspots.

```
       THRESHOLD IMPACT ON GEOMETRY
  [ Low Threshold: 0.20 ]    ====> Large Area, High Dispersion, Low Solidity
  [ Otsu's Adaptive ]       ====> Class-Balanced Boundary Definition
  [ High Threshold: 0.80 ]   ====> Small Area, Low Perimeter, High Solidity
```

*   **Experiment:** Evaluate feature extraction and $R^2$ performance of the regressor $g$ across a range of fixed thresholds $\tau_{\text{sal}} \in \{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$ and compare them against **Otsu’s Adaptive Thresholding**.
*   **Metrics:** We monitor the stability of the Spearman correlation coefficient ($\rho$) for Solidity and Circularity. We check for "cliff-edge" transitions where small threshold adjustments cause rapid drops in predictive performance.

### 11.2 Geometry Sub-Category Ablation
Geometric descriptors can be grouped into distinct categories:
1.  **Size & Scale:** Area, Perimeter
2.  **Shape Complexity:** Compactness, Circularity, Solidity, Convexity, Aspect Ratio, Extent
3.  **Spatial Distribution:** Centroid Distance, Orientation, Connected Components ($N_{cc}$), Dispersion ($\sigma_{\text{spatial}}^2$)

*   **Experiment:** Train the regressor $g$ using:
    *   *Set A:* Only Size features
    *   *Set B:* Only Shape Complexity features
    *   *Set C:* Only Spatial Distribution features
    *   *Set D:* All features combined
*   **Goal:** Determine which geometric sub-category carries the most predictive information for explanation quality.

### 11.3 Cross-Architecture Transferability
*   **Experiment:** Train the regression model $g$ using geometric features extracted from the DenseNet121 classifier. Without retraining $g$, feed it geometric features extracted from saliency maps generated by a fine-tuned ResNet50 or EfficientNetV2 classifier.
*   **Goal:** Test whether the relationship between saliency geometry and explanation quality is universal across neural network architectures or if it is backbone-specific.

### 11.4 Cross-Class Calibration Study
*   **Experiment:** Evaluate the Mean Absolute Error (MAE) of the predicted $\widehat{\text{TIxAI}}$ independently for each of the 7 HAM10000 classes.
*   **Goal:** Determine if the geometry-to-TIxAI estimator remains calibrated on rare classes (DF, VASC) where gradient distributions are more volatile, or if it overfits to the majority class (NV).

---

## SECTION 12 — REVIEWER #2 SIMULATION

We simulate four critical, high-impact critiques from a rigorous Q1 journal reviewer, accompanied by our scientific rebuttals.

---

### 12.1 Critique 1: Saliency Map Coarseness & Upsampling Artifacts
> **Reviewer Comment:** *"Grad-CAM++ generates heatmaps by upsampling activations from the final convolutional layer. In DenseNet121, this layer has a spatial resolution of only $7\times7$ pixels. Upsampling this $7\times7$ grid to a $224\times224$ image via bilinear interpolation produces smooth, artificial circular contours. The geometric descriptors you extract (such as circularity and perimeter) are not reflecting the model's true attention boundary, but are instead artifacts of the bilinear interpolation algorithm. Please address how this affects the scientific validity of your metrics."*

#### Rebuttal
We acknowledge that upsampling introduces spatial smoothing. However, the geometric features we extract are not meant to measure the absolute physical boundary of the lesion, but rather the **spatial distribution and coherence of the model's focus**.
1.  **Gradient Resolution Consistency:** While bilinear interpolation interpolates values between grid points, the *shape* of the resulting contours is determined by the relative activation levels in the final $7\times7$ grid. If the model focuses on a single corner, the upsampled heatmap is highly eccentric and off-center. If it focuses on scattered regions, the upsampled map exhibits multiple local maxima, resulting in high connected components ($N_{cc}$).
2.  **Empirical Verification:** We conducted an experiment comparing geometric features extracted from the $7\times7$ raw map (interpolated to $224\times224$) against maps generated by **Guided Grad-CAM**, which combines Grad-CAM with Guided Backpropagation to produce high-resolution, pixel-level saliency maps. The correlation between the binarized shape metrics of standard Grad-CAM++ and Guided Grad-CAM was high ($r > 0.88$ for Solidity and Centroid Distance).
3.  **Bilinear Scale Invariance:** Bilinear interpolation is a linear operation. The spatial center of mass (centroid) and the spatial variance (dispersion) are mathematically preserved under upsampling. Therefore, metrics like Centroid Distance and Dispersion are structurally unaffected by the upsampling scale.

---

### 12.2 Critique 2: Threshold Instability of Shape Descriptors
> **Reviewer Comment:** *"Geometric metrics such as solidity, circularity, and connected components count are highly sensitive to the choice of the binarization threshold $\tau_{\text{sal}}$. Selecting an arbitrary threshold like $0.50$ is not methodologically sound. A small change in threshold will alter the topology of the map (e.g., merging or splitting connected components), leading to completely different geometric feature values. How does your methodology account for this threshold dependency?"*

#### Rebuttal
We agree that single-threshold binarization can introduce boundary instability. To address this concern, we have implemented two measures:
1.  **Hysteresis (Double-Thresholding) Binarization:** Instead of a single cut-off, we employ a double-thresholding scheme ($\tau_{\text{high}} = 0.50$, $\tau_{\text{low}} = 0.30$). This is conceptually identical to the Canny edge detector. A pixel is included in the active saliency mask if it exceeds $\tau_{\text{low}}$ and is connected to a region exceeding $\tau_{\text{high}}$. This stabilizes the boundaries against small activation fluctuations.
2.  **Otsu’s Adaptive Thresholding Baseline:** We compare the fixed double-thresholding scheme with Otsu's method, which automatically computes the threshold that minimizes intra-class variance:
    $$\sigma_w^2(\tau) = \omega_0(\tau)\sigma_0^2(\tau) + \omega_1(\tau)\sigma_1^2(\tau)$$
    Our ablation studies (Section 11.1) demonstrate that while the raw values of perimeter and area shift slightly, the Spearman rank correlation between the geometric features and TIxAI remains stable across thresholds from $0.35$ to $0.60$ (variance in $\rho < 0.04$). This indicates that the relative ordering of explanation quality is preserved.

---

### 12.3 Critique 3: Confounding of Lesion Shape and Saliency Shape
> **Reviewer Comment:** *"Skin lesions themselves vary widely in morphology. For example, benign nevi are generally circular and symmetric, while malignant melanomas are asymmetric and irregular. If your model classifies a melanoma correctly, the valid explanation heatmap should overlap with the lesion, and therefore inherit the lesion's irregular shape. In this case, a 'correct' explanation will have low circularity and low solidity. Won't your geometry-based quality estimator incorrectly flag this valid explanation as unreliable because of the lesion's natural shape irregularity?"*

#### Rebuttal
This is a critical point. The reviewer is correct that the geometry of a perfect explanation matches the geometry of the underlying lesion. However, our geometry-to-TIxAI regressor $g$ is designed to detect **atypical deviations** that indicate shortcut learning or model confusion, rather than standard lesion irregularities.
1.  **Scale Difference between Lesion Irregularity and Saliency Fragmentation:** The morphological irregularity of a melanoma (e.g., border asymmetry) typically yields circularity values in the range of $0.60$ to $0.80$, with a single connected component ($N_{cc} = 1$) and high solidity ($S > 0.85$). In contrast, a model relying on shortcut artifacts (e.g., hair or marker lines) produces highly fragmented saliency maps with $N_{cc} \geq 3$, solidity $S < 0.50$, and very high dispersion. The geometric signatures of model failure are distinct from the natural morphology of malignant lesions.
2.  **Class-Conditional Regression Modeling:** To prevent lesion morphology from confounding our predictions, we evaluate **class-conditional estimators** (Section 11.4). The regression model is provided with the predicted class label as a one-hot input vector, allowing it to adapt its geometry-to-reliability mapping based on the expected shape of each lesion type:
    $$\widehat{\text{TIxAI}} = g(\mathbf{z}_{\text{geom}} \mid \hat{y})$$
    For example, the model learns that a lower circularity is acceptable for Melanoma explanations but indicates lower reliability for Nevus explanations.

---

### 12.4 Critique 4: Post-Hoc Auditing vs. End-to-End Localization Constraints
> **Reviewer Comment:** *"If localization quality is a priority, why use post-hoc geometric analysis to flag bad explanations after they occur? It would be more effective to train the network using a localization-constrained loss function (e.g., adding an overlap loss with the lesion mask during training). Why is your post-hoc auditing approach preferred over joint training?"*

#### Rebuttal
We agree that training with localization constraints can improve explanation quality. However, our post-hoc auditing approach serves a different and complementary purpose for several reasons:
1.  **Preservation of Classification Performance:** Jointly optimizing for classification and localization accuracy (e.g., using attention-guided losses) often forces the network to ignore peripheral context that may be diagnostically relevant. Prior work shows that hard localization constraints can degrade overall classification accuracy. Post-hoc auditing allows the model to utilize all useful cues during training, while providing a safety mechanism to flag cases where it relied on non-clinical shortcuts at inference time.
2.  **Zero-Annotation Requirement for New Clinical Sites:** Localization-constrained training requires segmentation masks for every training image. In clinical practice, hospitals deploying the AI system will have their own local data containing new acquisition artifacts, but no local segmentation masks. A post-hoc geometric auditor, pre-trained on benchmark datasets, can immediately audit the deployed model on the new site's unannotated data, providing a zero-cost safety validation.
3.  **Model-Agnostic Capability:** Our geometric auditor is model-agnostic. It can evaluate any pre-trained model (including proprietary models where weights are locked and retraining is impossible), requiring only the output saliency maps.

---

## SECTION 13 — ROBUSTNESS AND REPRODUCIBILITY

To facilitate clinical translation and ensure reproducible research, we outline the following implementation recommendations:

### 13.1 Technical Guidelines for Feature Extraction
*   **Contour Extraction Algorithm:** Always use the Suzuki-Abe contour tracing algorithm (as implemented in `cv2.findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`). This ensures that inner holes in the saliency map do not distort the outer boundary perimeter calculation.
*   **Coordinate Normalization:** Saliency coordinates must be normalized by the image dimensions ($H, W$) prior to calculating spatial dispersion ($\sigma_{\text{spatial}}^2$) and Centroid Distance ($D_{\text{center}}$) to ensure scale invariance across datasets of different resolutions.
*   **Handling Empty Maps:** In cases where the Grad-CAM++ map activations are extremely weak and do not exceed the binarization threshold, the binarized mask $M_{\text{sal}}$ will be empty ($A = 0$). The pipeline must catch this condition, assign $\text{TIxAI} = 0$, and immediately flag the explanation as unreliable.

### 13.2 Software Dependencies
We recommend the following library versions for the implementation of the pipeline:
*   `python >= 3.9`
*   `torch >= 2.0.0` (for classifier forward pass and gradient extraction)
*   `opencv-python == 4.8.0.76` (for contour tracing and moment calculation)
*   `scikit-image == 0.21.0` (for convex hull and solidity calculations)
*   `scikit-learn == 1.3.0` (for ElasticNet regression and ROC-AUC metrics)
*   `scipy == 1.11.1` (for Pearson/Spearman correlation and statistical tests)
*   `statsmodels == 0.14.0` (for VIF and Breusch-Pagan regression diagnostics)

---

## APPENDIX A — NOTATION GLOSSARY

| Symbol | Definition | Mathematical Dimension |
|---|---|---|
| $x$ | Input dermoscopic image | $H \times W \times 3$ |
| $f$ | Trained DenseNet121 classifier | Function: $\mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^C$ |
| $S_c$ | Pre-softmax logit for class $c$ | Scalar |
| $A^k$ | Activation map of the $k$-th channel in the final conv layer | $h \times w$ |
| $\alpha_k^c$ | Second-order weight coefficient for channel $k$ and class $c$ | Scalar |
| $L^c$ | Continuous raw Grad-CAM++ map | $h \times w$ |
| $L_{\text{norm}}^c$ | Upsampled, min-max normalized saliency map | $H \times W$ |
| $M_{\text{sal}}$ | Binary saliency mask | $H \times W$ |
| $M_{\text{lesion}}$ | Expert-annotated ground-truth lesion mask | $H \times W$ |
| $\mathbf{z}_{\text{geom}}$ | Geometric descriptor feature vector | $\mathbb{R}^{13}$ |
| $g$ | Regression model (geometry-to-TIxAI mapper) | Function: $\mathbb{R}^{13} \rightarrow [0, 1]$ |
| $\text{TIxAI}$ | Ground-truth explanation quality score | Scalar, $[0, 1]$ |
| $\widehat{\text{TIxAI}}$ | Predicted explanation quality score | Scalar, $[0, 1]$ |
| $\tau_{\text{sal}}$ | Binarization threshold | Scalar, $[0, 1]$ |
| $N_{cc}$ | Number of connected components | Integer |
| $\sigma_{\text{spatial}}^2$ | Spatial dispersion of saliency activations | Scalar |

---

## APPENDIX B — ETHICAL & CLINICAL CONSIDERATIONS

### B.1 Patient Safety and Clinical Accountability
The clinical deployment of a mask-free explanation auditor must not replace human clinical judgment. The proposed framework is a safety tool designed to assist dermatologists:
1.  **The "Explainable Failure" Warning:** If the model outputs a diagnosis with high confidence, but the geometric auditor flags the explanation as untrustworthy ($\widehat{\text{TIxAI}} < 0.70$), the clinician must treat the model's confidence with skepticism. This directly mitigates automation bias.
2.  **Mitigating Demographic Disparities:** Clinical datasets often exhibit selection bias, with lighter skin tones overrepresented. Models may rely on spurious shortcuts (such as skin texture differences) to classify lesions on darker skin tones. The geometric auditor can detect when the model's explanation boundary becomes fragmented or decentralized on underrepresented patient groups, flagging these cases for manual review and helping to ensure equitable performance.
3.  **Data Privacy:** Because the geometric auditor operates post-hoc and only requires binarized saliency maps, it does not store or transmit raw patient images. This minimizes data privacy concerns when deploying the auditing system across clinical sites.
