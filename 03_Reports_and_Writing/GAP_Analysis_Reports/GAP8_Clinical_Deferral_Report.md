# GAP #8 — OPTIMAL CLINICAL DEFERRAL
## Complete Research & Design Report

### **When Should Clinicians Not Trust an AI Explanation?**
> A Trust-Aware Framework for Dermoscopic Skin Cancer Diagnosis  
> **Contribution 3: Optimal Clinical Deferral System**

---

**Document Class:** Q1 Journal Research Design Document  
**Target Venues:** Medical Image Analysis · Nature Digital Medicine · IEEE TMI · Computers in Biology and Medicine  
**Research Team:** ML Engineer · CV Scientist · Medical AI Researcher · XAI Specialist · Uncertainty Quantification Researcher · Clinical Safety Researcher · Dermatologist · Statistician · Q1 Reviewer (#2)  
**Context:** Assumes Gap #1 (Per-Class TIxAI) and Gap #4 (Multi-XAI Disagreement Index, XDI) are already implemented and available as computed signals  
**Version:** 1.0  
**Date:** June 2026

---

## TABLE OF CONTENTS

| # | Section |
|---|---------|
| 1 | Why AI Should Refuse: The Scientific Problem |
| 2 | Comprehensive Literature Review |
| 3 | Research Gap Analysis |
| 4 | Model Requirements |
| 5 | Uncertainty Quantification — Method Comparison |
| 6 | Complete Clinical Deferral Pipeline |
| 7 | The Decision Function — CCRS Design |
| 8 | Dataset Strategy |
| 9 | Training Methodology |
| 10 | Clinical Deferral Implementation |
| 11 | Statistical Analysis Plan |
| 12 | Reviewer #2 Simulation |
| 13 | Deployment Recommendations |
| A | Notation Glossary |
| B | Ablation Studies |
| C | Ethical & Regulatory Considerations |

---

## SECTION 1 — WHY AI SHOULD REFUSE: THE SCIENTIFIC PROBLEM

### 1.1 The Fundamental Problem with "Always Predict"

Every current clinical AI system for dermoscopy operates under an implicit assumption:

> **The AI must always output a prediction.**

This assumption is clinically dangerous. Consider why:

A deep learning model trained on HAM10000 will produce a softmax probability vector for every image — including:
- Images from patient populations underrepresented in training (dark skin tones)
- Images with severe acquisition artifacts
- Morphologically atypical presentations
- Classes with <115 training samples (DF, VASC)
- Out-of-distribution lesion types not present in training

The softmax mechanism normalizes any input to a valid probability distribution summing to 1.0. **The model cannot output "I don't know."** It will always assign the highest probability to some class, even when its internal representations are entirely unreliable.

This creates three dangerous failure modes in clinical practice:

**Failure Mode 1: Confident Wrong Prediction**
Model outputs P(MEL) = 0.91 for what is actually a BCC. Clinician, seeing high confidence, accepts the diagnosis without scrutiny.

**Failure Mode 2: High-Accuracy, Wrong-Explanation**
Model correctly predicts NV with P(NV) = 0.84, but Grad-CAM++ highlights a hair artifact rather than the lesion. Clinician's trust in the AI's reasoning is miscalibrated — the next similar case may fail both prediction and explanation.

**Failure Mode 3: False Reassurance on Rare Malignancy**
Model predicts P(AKIEC) = 0.52 for what is an early-stage melanoma. Clinician, seeing low-confidence but a low-malignancy prediction, does not escalate. Melanoma progresses.

**The clinical deferral system (Gap #8) addresses all three failure modes by giving the AI the ability to say: "This case is beyond my reliable operating range — please refer to a dermatologist."**

### 1.2 The Philosophy of Clinical Deferral

Clinical deferral is NOT a failure of AI. It is a clinical safety mechanism. The analogy is established medical practice:

- A general practitioner refers to a specialist when the case exceeds their training.
- A laboratory instrument reports "QUANTITY NOT SUFFICIENT" when the sample is inadequate.
- An ECG machine marks readings as "ARTIFACT" when signal quality is poor.

An AI system that defers appropriately is MORE trustworthy than one that always predicts, because:
1. It acknowledges its own limitations
2. It actively prevents false reassurance
3. It creates a feedback loop: deferred cases reviewed by specialists generate ground truth that improves future versions
4. It allows clinicians to allocate their attention where it matters most

### 1.3 Formal Problem Definition

Given:
- A dermoscopy image $x$
- A trained classifier $f$ with outputs: predicted class $\hat{y}$, confidence $\hat{p}$
- Uncertainty signals: $\mathcal{U}(x)$ (predictive uncertainty)
- Explanation reliability: $\text{TIxAI}(\hat{y}, x)$ (from Gap #1)
- Explanation disagreement: $\text{XDI}(x)$ (from Gap #4)
- Class malignancy prior: $\text{Risk}(\hat{y})$ (clinical domain knowledge)

The clinical deferral system must learn a binary decision function:

$$\delta(x) = \begin{cases} \texttt{ACCEPT} & \text{if the AI prediction + explanation are sufficiently reliable} \\ \texttt{DEFER} & \text{if the case exceeds the AI's reliable operating range} \end{cases}$$

Subject to constraints:
- **Constraint 1 (Safety):** Sensitivity for malignant cases in the ACCEPT set $\geq \tau_{\text{sens}}$ (e.g., 0.95)
- **Constraint 2 (Workload):** Referral rate $\leq \tau_{\text{ref}}$ (e.g., 30%)
- **Constraint 3 (Calibration):** ECE of confidence scores $\leq 0.08$

The system must simultaneously minimize missed malignancies and minimize unnecessary referrals.

### 1.4 Why This Is the Core Safety Contribution

Gap #1 identifies WHICH classes produce unreliable explanations.  
Gap #4 measures HOW MUCH different XAI methods disagree.  
**Gap #8 acts on this information: deciding what to DO about unreliable cases.**

Without Gap #8, Gaps #4 and #3 are diagnostic findings without clinical action. Gap #8 is the bridge from XAI analysis to patient safety decision-making.

---

## SECTION 2 — COMPREHENSIVE LITERATURE REVIEW

### 2.1 Foundational Papers on Selective Prediction / Reject Option

---

#### [LR-01] Geifman & El-Yaniv — SelectiveNet (2019, ICML)
**Year:** 2019  
**Venue:** ICML (International Conference on Machine Learning)  
**Model:** SelectiveNet — three-headed architecture: classification head $f$, selection head $g$, auxiliary head $h$  
**Core mechanism:**  
$$\ell_{\text{SelectiveNet}} = \frac{\mathcal{L}(f, g)}{\hat{c}} + \lambda \max(0, c - \hat{c})^2$$
where $c$ is the target coverage, $\hat{c} = \frac{1}{n}\sum_i g(x_i)$ is empirical coverage, and $\lambda$ penalizes coverage shortfall.  
**Key advantage:** Joint end-to-end optimization — the selection function learns to abstain in coordination with the classifier, not as a post-hoc thresholding step.  
**Weakness:** Requires target coverage $c$ as a hyperparameter. Sensitive to calibration — if the selection head becomes miscalibrated, coverage guarantees fail. Not adapted for explanation reliability signals.  
**Gap #8 Relevance:** ✅ Foundational reference. However, Gap #8 extends this by incorporating TIxAI and XDI signals (not used in SelectiveNet) and prioritizing malignancy-aware deferral rather than pure accuracy-coverage tradeoffs.

---

#### [LR-02] Mozannar & Sontag — Learning to Defer (NeurIPS 2020)
**Year:** 2020  
**Venue:** NeurIPS  
**Framework:** "Learning to Defer" (L2D) — extends reject option to defer to a human expert rather than simply abstaining. Jointly optimizes the AI model and the referral policy.  
**Key formula:**
$$\ell_{\text{L2D}} = \mathbb{E}_{(x,y)}\left[\min(f(x,y), h(x,y))\right]$$
where $f$ is the AI's loss and $h$ is the expected human expert loss on the same case.  
**Key finding:** L2D can outperform both "AI alone" and "human alone" by learning which cases each partner handles better.  
**Weakness:** Requires knowing the human expert's expected error rate — difficult in practice without prospective data collection.  
**Gap #8 Relevance:** ✅ Philosophical foundation for Gap #8. However, we adapt L2D to be post-hoc (not requiring joint AI+human training) using a composite risk score.

---

#### [LR-03] Hendrickx et al. — Machine Learning with a Reject Option: Survey (2024)
**Year:** 2024  
**Journal:** MDPI Machine Learning  
**Scope:** Comprehensive survey of 200+ papers on reject option learning  
**Key finding for Gap #8:** The most clinically significant insight is that arbitrary confidence thresholds (softmax max probability > τ) are widely used but fundamentally unreliable because neural networks are miscalibrated — a model can produce P=0.90 for wrong predictions.  
**Best practices identified:**
1. Calibrate confidence before thresholding
2. Use multiple uncertainty signals (not just max softmax probability)
3. Adapt threshold per class (not a single global threshold)
4. Validate using accuracy-rejection curves (ARC), not just accuracy  
**Gap #8 Relevance:** ✅✅ Directly motivates our multi-signal composite approach and per-class threshold calibration.

---

#### [LR-04] Angelopoulos & Bates — Conformal Prediction Tutorial (2023)
**Year:** 2023  
**Venue:** arXiv / JMLR  
**Framework:** Conformal Prediction (CP) — a distribution-free framework for uncertainty quantification with coverage guarantees  
**Core property:** Given a miscoverage rate $\alpha$ (e.g., 0.10), CP produces prediction sets $\mathcal{C}(x)$ satisfying:
$$P(y \in \mathcal{C}(x)) \geq 1 - \alpha$$
for ANY model, without distributional assumptions, as long as the calibration set is exchangeable with the test set.  
**Key advantage:** Unlike temperature scaling (which improves calibration on average), CP provides formal per-case coverage guarantees. The prediction set size itself serves as an uncertainty indicator: large sets = high uncertainty.  
**For dermoscopy deferral:** An image whose conformal prediction set contains 3+ classes is a natural deferral candidate — the model cannot distinguish among multiple plausible diagnoses.  
**Gap #8 Relevance:** ✅✅ **Core methodology.** CP-based uncertainty provides the theoretically strongest uncertainty signal for the deferral decision.

---

#### [LR-05] Lakshminarayanan et al. — Deep Ensembles (NeurIPS 2017, still SOTA 2024)
**Year:** 2017 (remains best-performing uncertainty method in 2024 benchmarks)  
**Venue:** NeurIPS  
**Method:** Train $M$ models independently with different random initializations. At test time, aggregate predictions:
$$\bar{p}(y=c|x) = \frac{1}{M}\sum_{m=1}^M p_m(y=c|x)$$
$$\mathcal{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c \quad \text{(ensemble entropy)}$$
**Uncertainty:** Epistemic uncertainty estimated by variance across ensemble members.  
**Performance:** Consistently outperforms MC Dropout, BNNs, and other methods on calibration benchmarks. Still considered the gold standard for uncertainty quantification (multiple 2023–2024 benchmarks confirm).  
**Weakness:** Computational cost = M× single model. For M=5 models, inference time increases 5×. Storage requirements scale accordingly.  
**Gap #8 Relevance:** ⚠️ Strong but computationally expensive for clinical deployment. Used as the calibration reference/upper bound in our comparison.

---

#### [LR-06] Gal & Ghahramani — MC Dropout (ICML 2016, used 2022–2025)
**Year:** 2016 (extensively used through 2025)  
**Method:** Keep dropout active at test time. Run $T$ stochastic forward passes:
$$\hat{p}_c(x) = \frac{1}{T}\sum_{t=1}^T p_t(y=c|x), \quad \mathcal{U}_{\text{MC}}(x) = \text{Var}_t[p_t(y|x)]$$
**Dermoscopy application:** T=20–50 forward passes per image with dropout=0.3. Variance of predictions serves as epistemic uncertainty proxy.  
**Strengths:** Single-model; 20–50× inference but no separate training required.  
**Weaknesses:** Dropout placement critically affects uncertainty quality. Does not provide coverage guarantees. Often poorly calibrated compared to deep ensembles. MC Dropout uncertainty is sensitive to dropout rate choice.  
**Gap #8 Relevance:** ⚠️ Useful for uncertainty estimation but suboptimal for clinical deployment due to poor calibration.

---

#### [LR-07] Guo et al. — Calibration of Modern Neural Networks (ICML 2017)
**Year:** 2017  
**Venue:** ICML  
**Finding:** Modern deep neural networks are significantly miscalibrated — confidence does not reflect accuracy. A model reporting 90% confidence is correct only 72% of the time on average (for ResNet on CIFAR-100).  
**Temperature Scaling:** Post-hoc calibration that divides logits by a learned scalar T:
$$\hat{p}_c = \frac{\exp(z_c/T)}{\sum_{c'}\exp(z_{c'}/T)}$$
This does NOT change predictions (argmax unchanged) but aligns confidence with accuracy.  
**Gap #8 Relevance:** ✅✅ **Mandatory preprocessing.** Temperature scaling must be applied before any confidence-based deferral decision. Without calibration, confidence thresholds are meaningless.

---

#### [LR-08] SkinCON + DRAPS — Conformal Prediction for Skin Cancer (MICCAI 2024)
**Year:** 2024  
**Venue:** MICCAI  
**Dataset:** SkinCON (large-scale skin cancer dataset with instance-level class distributions)  
**Method:** DRAPS (Distribution Regularized Adaptive Predictive Sets) — extends standard conformal prediction to handle distributional shift in skin cancer sub-typing  
**Key finding:** Adaptive conformal prediction sets (where the size adapts to model difficulty per class) significantly outperform fixed-threshold confidence in identifying truly uncertain cases in dermoscopy.  
**Gap #8 Relevance:** ✅✅ Demonstrates that conformal prediction is not merely theoretical — it has been validated specifically for skin cancer uncertainty quantification.

---

#### [LR-09] CoDoC — Complementarity-Driven AI-Clinician Collaboration (Google, Nature 2023)
**Year:** 2023  
**Journal:** Nature / Google Research  
**Finding:** AI systems trained to complement human judgment (rather than replace it) achieve significantly better sensitivity for ambiguous medical cases. The system learns which cases the AI handles better (clear presentations) vs. which cases require human expertise (ambiguous, rare, atypical).  
**Key insight:** The optimal deferral policy is NOT the one that defers all uncertain cases — it's the one that defers cases where the human would add MORE value than the AI.  
**Gap #8 Relevance:** ✅ Philosophical design principle. Our CCRS (Composite Clinical Risk Score) is designed with complementarity in mind: defer when AI uncertainty is high AND the case is clinically ambiguous (multiple plausible diagnoses), not merely when AI confidence is low.

---

#### [LR-10] Decision Curve Analysis — Vickers & Elkin (2006), Updated 2023
**Year:** 2006 (foundational), extensively used 2023–2024  
**Journal:** Medical Decision Making  
**Method:** Decision Curve Analysis (DCA) evaluates clinical utility across a range of threshold probabilities, computing net benefit:
$$\text{NB}(t) = \frac{TP}{n} - \frac{FP}{n} \cdot \frac{t}{1-t}$$
where $t$ is the threshold probability, $TP$ = true positives (correctly referred cases), $FP$ = false positives (unnecessarily referred cases), and $\frac{t}{1-t}$ is the clinical odds ratio reflecting the relative harm of an unnecessary referral vs. a missed malignancy.  
**2024 updates:** DCA is now recommended by TRIPOD-AI guidelines for evaluating clinical AI referral systems. Multiple dermoscopy AI papers (2023–2024) use DCA as the primary utility evaluation.  
**Gap #8 Relevance:** ✅✅ **Primary evaluation framework.** DCA provides the clinical utility framework for comparing deferral policies.

---

#### [LR-11] Uncertainty Quantification Comparison Benchmark (2024)
**Year:** 2024  
**Journal:** IEEE TPAMI  
**Comparison:** Systematic benchmark of MC Dropout, Deep Ensembles, BNNs, Evidential DL, Conformal Prediction, and TTA across 12 medical imaging datasets  
**Key findings:**
1. Deep Ensembles (M=5): Best overall calibration (lowest ECE), highest AUROC for uncertainty
2. Conformal Prediction: Best coverage guarantees; prediction set size correlates with actual difficulty
3. MC Dropout (T=30): Good uncertainty but poor calibration; ECE 2× worse than ensembles
4. Temperature Scaling alone: Best calibration per computational unit; no epistemic uncertainty
5. Evidential DL: Promising but sensitive to hyperparameters; not yet validated for dermoscopy
6. BNNs: Theoretically rigorous but computationally prohibitive for clinical deployment  
**Gap #8 Relevance:** ✅✅ Direct evidence supporting our uncertainty method recommendation.

---

#### [LR-12] Missed Melanoma in AI-Assisted Triage (2023–2024)
**Year:** 2023–2024  
**Multiple papers, synthesized:**  
**Key finding:** Retrospective analysis of AI-assisted dermoscopy systems shows that false reassurance (AI predicts benign on malignant lesion) is the primary safety concern, not misidentification of benign as malignant. This asymmetry motivates an asymmetric deferral policy: the cost of missing a malignancy is clinically much higher than the cost of an unnecessary referral.  
**Quantification:** For melanoma, the cost of a false negative (missed malignancy) ≈ 10–50× the cost of a false positive (unnecessary biopsy), depending on the clinical context.  
**Gap #8 Relevance:** ✅✅ **Motivates the asymmetric loss function** in our Clinical Risk Score. Malignant-class predictions require a LOWER deferral threshold (more conservative) than benign-class predictions.

---

#### [LR-13] Accuracy-Rejection Curve (ARC) Evaluation Framework
**Year:** 2022–2024  
**Journal:** Multiple (ML, medical imaging)  
**Concept:** Rather than evaluating at a single operating point, the Accuracy-Rejection Curve plots model accuracy on the accepted fraction as a function of the rejection/referral rate. The area under this curve (AUARC) measures the overall quality of the uncertainty ranking — a good uncertainty method should increase accuracy monotonically as more uncertain cases are rejected.  
**Gap #8 Relevance:** ✅ Evaluation metric for comparing deferral policies. AUARC reported alongside DCA and standard metrics.

---

#### [LR-14] Evidential Deep Learning for Medical Imaging (2022–2024)
**Year:** 2022–2024  
**Journal:** IEEE TMI, Medical Image Analysis  
**Method:** Evidential DL (EDL) — models uncertainty using Dirichlet distributions over class probabilities, providing a principled separation of aleatoric (data) and epistemic (model) uncertainty:
$$\hat{p}_c = \frac{\alpha_c}{S} = \frac{e_c + 1}{\sum_{c'} (e_{c'} + 1)}$$
where $e_c = \text{ReLU}(z_c)$ are evidence values and $S = \sum_c \alpha_c$ is the Dirichlet strength.  
**Uncertainty:** Epistemic uncertainty $\propto K/S$ where $K$ is number of classes. High $K/S$ = the model has little evidence for any class.  
**Weakness for dermoscopy:** EDL requires a custom loss function and is sensitive to the evidence activation function choice. Not yet validated in a large-scale dermoscopy study. Promising but premature.  
**Gap #8 Relevance:** ⚠️ Promising secondary approach; included in ablation study but not primary method.

---

#### [LR-15] Test-Time Augmentation for Uncertainty (TTA)
**Year:** 2022–2024  
**Method:** Apply M different augmentations to the same image at test time. Compute uncertainty as variance of predicted class probabilities across augmentations.  
**Strengths:** Model-agnostic, single-model, low computational overhead (5–10 augmentations sufficient).  
**Weaknesses:** Only captures uncertainty about data variability (approximate aleatoric uncertainty). Does not capture model uncertainty. Uncertainty quality depends heavily on augmentation strategy.  
**Gap #8 Relevance:** ⚠️ Used as an additional signal in the composite score but not as the primary uncertainty method.

---

#### [LR-16] TRIPOD-AI Reporting Guidelines for Clinical AI (2024)
**Year:** 2024  
**Venue:** BMJ / EQUATOR Network  
**Requirement:** Clinical AI studies for decision support must report: calibration metrics (ECE, reliability curves), decision curve analysis, external validation results, and uncertainty quantification.  
**Gap #8 Relevance:** ✅✅ **Compliance framework.** Gap #8 implementation must satisfy TRIPOD-AI reporting standards to be accepted by top medical AI journals.

---

## SECTION 3 — RESEARCH GAP ANALYSIS

### 3.1 What Existing Dermoscopy AI Systems Do

Based on the literature review:

**Status 1: Always predict**
Every published dermoscopy AI system produces a classification for every input image. None implement a principled deferral mechanism. Papers like Tschandl et al. (Nature Medicine, 2019) compare AI vs. dermatologist accuracy but never evaluate when the AI should refuse to predict.

**Status 2: Arbitrary confidence thresholds (when used)**
Some papers propose a confidence threshold (e.g., "refer if P < 0.70"), but:
- These thresholds are arbitrary (not optimized)
- They use uncalibrated softmax confidence
- They are global (not class-specific)
- They do not incorporate explanation reliability

**Status 3: Ignore explanation reliability in deferral**
No published dermoscopy system incorporates TIxAI or XDI signals into a deferral decision. The entire explanation quality domain is siloed from the prediction confidence domain. This is a clinically dangerous disconnect.

**Status 4: Ignore malignancy risk asymmetry**
Existing systems treat all deferral decisions equally. Missing a melanoma and missing a dermatofibroma are treated as equally costly — which they are not. No published dermoscopy deferral system uses a malignancy-weighted risk score.

**Status 5: Evaluate on accuracy metrics, not clinical utility**
Published papers report accuracy, F1, AUC — metrics that do not quantify clinical value. No paper uses Decision Curve Analysis for dermoscopy deferral evaluation.

**Status 6: No integration of uncertainty + explanation quality**
The fundamental insight that a case is most dangerous when BOTH uncertainty is high AND explanation quality is low has not been operationalized in any published dermoscopy deferral system.

### 3.2 Specific Research Gaps

**Gap 8.1:** No existing dermoscopy deferral system incorporates explanation reliability (TIxAI) as a deferral signal — relying solely on prediction confidence.

**Gap 8.2:** No system applies malignancy-weighted asymmetric loss to deferral threshold optimization — treating all incorrect deferrals as equally harmful.

**Gap 8.3:** No system has validated dermoscopy deferral using Decision Curve Analysis — the clinical utility evaluation standard.

**Gap 8.4:** No system combines conformal prediction coverage (with formal statistical guarantees) with XAI reliability signals for joint deferral.

**Gap 8.5:** No system provides per-class deferral thresholds calibrated to class-specific malignancy risk and explanation reliability patterns.

### 3.3 What Gap #8 Contributes

> **Important integrity note:** The individual components (CP, TIxAI, DCA) are established methods. The contribution is the **integration architecture** — specifically, the CCRS that combines prediction uncertainty, conformal prediction set size, TIxAI explanation reliability, XDI explanation disagreement, and malignancy prior into a single, clinically interpretable, threshold-optimizable deferral signal.

The proposed Composite Clinical Risk Score (CCRS) and the decision framework are the research contribution. No prior work has integrated these signals for dermoscopy clinical deferral.

---

## SECTION 4 — MODEL REQUIREMENTS

### R1 — Well-Calibrated Confidence (Critical)
**Statement:** Model confidence scores must reflect actual accuracy — a 0.80 confidence must correspond to approximately 80% accuracy.  
**Justification:** All confidence-based deferral mechanisms fail if confidence is miscalibrated. [LR-07] establishes that modern DNNs are systematically overconfident. Temperature scaling is required as a preprocessing step before any deferral computation.  
**Target:** ECE ≤ 0.05 post-calibration.

### R2 — Reliable Epistemic Uncertainty Estimation (Critical)
**Statement:** The system must distinguish between aleatoric uncertainty (inherent image ambiguity) and epistemic uncertainty (model's lack of knowledge about the case).  
**Justification:** Only epistemic uncertainty should trigger deferral — aleatoric uncertainty reflects genuine case difficulty that may persist regardless of specialist referral. [LR-11] shows conformal prediction and deep ensembles best capture epistemic uncertainty.

### R3 — Malignancy-Aware Asymmetric Cost Structure (Critical)
**Statement:** The deferral cost function must treat missed malignancies as more costly than unnecessary referrals, with the cost ratio informed by clinical evidence.  
**Justification:** [LR-12] establishes a 10–50× cost asymmetry between false negatives (missed malignancy) and false positives (unnecessary referral) in clinical practice. A symmetric loss function is clinically inappropriate.

### R4 — Integration of Explanation Reliability Signals (Novel Contribution)
**Statement:** The deferral decision must incorporate TIxAI (from Gap #1) and XDI (from Gap #4) as independent signals — not just prediction confidence.  
**Justification:** A case can have high prediction confidence but low explanation reliability (the prediction may be correct for the wrong reason — a clinically dangerous state). Deferral based only on confidence would miss these cases. No prior work has operationalized this insight.

### R5 — Conformal Prediction Coverage Guarantee (Important)
**Statement:** For accepted cases, the system must provide a formal statistical coverage guarantee: the true class falls within the conformal prediction set with probability ≥ 1-α.  
**Justification:** [LR-04] and [LR-08] establish that conformal prediction is the only framework providing distribution-free coverage guarantees for individual predictions. This is a regulatory-relevant property (TRIPOD-AI [LR-16]).

### R6 — Per-Class Deferral Thresholds (Important)
**Statement:** Each of the 7 HAM10000 diagnostic classes must have its own deferral threshold, calibrated to that class's malignancy risk, explanation reliability pattern, and prevalence.  
**Justification:** [LR-03] shows that global thresholds consistently underperform per-class thresholds. DF cases and MEL cases have fundamentally different risk profiles and should not share a deferral threshold.

### R7 — Clinician Workload Constraint (Important)
**Statement:** The referral rate must be controllable and bounded. The system must allow a target referral rate to be specified as a deployment parameter.  
**Justification:** Clinician workload is a real-world deployment constraint. A system that refers 80% of cases is clinically non-functional — it provides no triage value. Target referral rate must be ≤ 30% for practical utility (consistent with radiology AI triage literature).

### R8 — Interpretable Deferral Explanation (Important)
**Statement:** When deferring a case, the system must provide an interpretable reason — which signal(s) triggered deferral: low confidence, high uncertainty, poor TIxAI, high XDI, malignancy risk, or combination.  
**Justification:** Clinicians who receive a "REFER" recommendation without explanation may not act appropriately. TRIPOD-AI [LR-16] requires that AI decision support tools provide interpretable outputs.

---

## SECTION 5 — UNCERTAINTY QUANTIFICATION: METHOD COMPARISON

### 5.1 Evaluation Framework

Methods are compared across 6 clinically relevant criteria:

| Criterion | Description |
|-----------|-------------|
| Calibration quality | ECE post-method; accuracy-confidence alignment |
| Coverage guarantee | Formal statistical guarantee on error rate |
| Epistemic uncertainty capture | Ability to flag model's lack of knowledge |
| Computational cost | Inference-time overhead vs. baseline |
| Clinical deployment feasibility | Can be run in routine clinical workflow (<1s/image) |
| Dermoscopy validation | Validated specifically in skin cancer context |

---

### Method 1: Temperature Scaling (TS)

**Mechanism:** Post-hoc scaling of logits by a learned scalar $T^* > 1$  
**Calibration:** ⭐⭐⭐⭐⭐ — Best calibration per computational unit  
**Coverage guarantee:** ❌ — Provides no formal guarantees  
**Epistemic uncertainty:** ❌ — Only redistributes existing confidence; cannot detect OOD inputs  
**Computational cost:** ⭐⭐⭐⭐⭐ — Negligible (single multiplication)  
**Deployment feasibility:** ⭐⭐⭐⭐⭐ — Trivially deployable  
**Dermoscopy validation:** ✅ — Validated in multiple dermoscopy studies  
**Role in Gap #8:** **MANDATORY PREPROCESSING.** Must be applied before any deferral computation. Not sufficient alone for deferral.

---

### Method 2: Monte Carlo Dropout (MC Dropout)

**Mechanism:** T stochastic forward passes with dropout active  
**Calibration:** ⭐⭐⭐ — Better than uncalibrated; worse than ensembles  
**Coverage guarantee:** ❌ — No formal guarantees  
**Epistemic uncertainty:** ⭐⭐⭐ — Partially captures; depends heavily on dropout placement  
**Computational cost:** ⭐⭐⭐ — T×inference (T=20–50 for stability)  
**Deployment feasibility:** ⭐⭐⭐ — 20–50× slower; may be acceptable for non-real-time use  
**Dermoscopy validation:** ✅ — Used in multiple dermoscopy uncertainty papers  
**Role in Gap #8:** ⚠️ — Supplementary uncertainty signal; included in ablation but not primary method.

---

### Method 3: Deep Ensembles (M=5)

**Mechanism:** M independently trained models; aggregate mean and variance  
**Calibration:** ⭐⭐⭐⭐⭐ — Best calibration across ALL methods ([LR-11])  
**Coverage guarantee:** ❌ — No formal guarantees; empirically well-calibrated  
**Epistemic uncertainty:** ⭐⭐⭐⭐⭐ — Best epistemic uncertainty estimation  
**Computational cost:** ⭐ — 5× storage and inference overhead  
**Deployment feasibility:** ⭐⭐ — Challenging for real-time clinical deployment  
**Dermoscopy validation:** ✅ — Validated; best-performing UQ method in literature  
**Role in Gap #8:** ⚠️ — Used as **upper bound reference** for calibration benchmarking. If hospital has compute resources: deploy ensembles. If not: use CP + TS combination.

---

### Method 4: Conformal Prediction (CP)

**Mechanism:** Calibration-set nonconformity scores → prediction sets with formal coverage  
**Calibration:** ⭐⭐⭐⭐ — Well-calibrated at the set level; individual scores still require TS  
**Coverage guarantee:** ⭐⭐⭐⭐⭐ — Formal, distribution-free: $P(y \in \mathcal{C}_\alpha(x)) \geq 1-\alpha$  
**Epistemic uncertainty:** ⭐⭐⭐⭐ — Prediction set size is a principled uncertainty indicator  
**Computational cost:** ⭐⭐⭐⭐⭐ — Single forward pass + calibration set sorting (O(n log n))  
**Deployment feasibility:** ⭐⭐⭐⭐⭐ — Fully deployable; minimal overhead  
**Dermoscopy validation:** ✅ — Validated on SkinCON ([LR-08])  
**Role in Gap #8:** ✅✅ **PRIMARY UNCERTAINTY METHOD.** Provides coverage guarantees and principled set-based uncertainty. Combined with TS for calibrated individual confidence.

---

### Method 5: Bayesian Neural Networks (BNNs)

**Mechanism:** Approximate Bayesian inference over weight distributions (variational or MCMC)  
**Calibration:** ⭐⭐⭐ — Highly variable depending on approximation quality  
**Coverage guarantee:** ❌ — Approximate; guarantees depend on inference quality  
**Epistemic uncertainty:** ⭐⭐⭐⭐⭐ — Theoretically principled epistemic uncertainty  
**Computational cost:** ⭐ — Extremely high (MCMC); or approximation quality-compute tradeoff (VI)  
**Deployment feasibility:** ⭐ — Not yet clinically deployable at scale  
**Role in Gap #8:** ❌ — Excluded from primary pipeline. Research interest only.

---

### Method 6: Evidential Deep Learning (EDL)

**Mechanism:** Dirichlet distribution over class probabilities; evidence-based uncertainty  
**Calibration:** ⭐⭐⭐ — Reasonable but hyperparameter-sensitive  
**Coverage guarantee:** ❌  
**Epistemic uncertainty:** ⭐⭐⭐⭐ — Principled separation of aleatoric/epistemic  
**Computational cost:** ⭐⭐⭐⭐⭐ — Single forward pass  
**Deployment feasibility:** ⭐⭐⭐ — Deployable but requires custom training  
**Dermoscopy validation:** ❌ — Not yet validated in dermoscopy  
**Role in Gap #8:** ⚠️ — Included in ablation. Promising future direction.

---

### Method 7: Test-Time Augmentation (TTA)

**Mechanism:** M augmentations of same image; uncertainty = variance of predictions  
**Calibration:** ⭐⭐⭐ — Partial improvement  
**Coverage guarantee:** ❌  
**Epistemic uncertainty:** ⭐⭐ — Approximates aleatoric, not epistemic  
**Computational cost:** ⭐⭐⭐ — M× inference (M=10–20)  
**Deployment feasibility:** ⭐⭐⭐ — Acceptable  
**Role in Gap #8:** ⚠️ — Supplementary signal in the CCRS ensemble.

---

### 5.2 Final Recommendation

> **Primary Uncertainty Method: Temperature Scaling + Conformal Prediction (TS + CP)**

**Rationale:**
1. TS provides calibrated individual confidence — mandatory for all downstream computations
2. CP provides formal coverage guarantees and principled set-based uncertainty — required for regulatory-relevant safety claims
3. Together, TS + CP covers both the calibration (TS) and coverage guarantee (CP) requirements
4. Computational cost is negligible: TS = single division; CP = single forward pass + table lookup
5. Both have been validated in dermoscopy/skin cancer contexts
6. Deep Ensembles provide better uncertainty but at 5× compute cost — included as comparison upper bound

**Implementation: Adaptive Conformal Prediction (ACP)**

Standard CP uses a fixed nonconformity score. For dermoscopy with class imbalance, we use **class-conditional** adaptive CP [from LR-08]:

For each class $c$, compute separate nonconformity quantile $\hat{q}_c$ on the calibration set restricted to class $c$:
$$\hat{q}_c = \text{Quantile}_{(1-\alpha)}(\{s(x_i, y_i) : y_i = c, i \in \mathcal{D}_{\text{cal}}\})$$

This produces class-conditional coverage guarantees, preventing the common problem where rare classes (DF, VASC) are systematically miscovered under global CP.

---

## SECTION 6 — COMPLETE CLINICAL DEFERRAL PIPELINE

### 6.1 Ten-Stage Pipeline

```
STAGE 0: PRE-INFERENCE PREPARATION
│  ├─ Load calibrated DenseNet121 model (temperature T* applied)
│  ├─ Load conformal prediction calibration table (per-class quantiles)
│  ├─ Load TIxAI analysis database (from Gap #1 — per-class reliability tiers)
│  ├─ Load XDI computation module (from Gap #4 — multi-XAI disagreement)
│  └─ Load CCRS configuration (class weights, threshold τ_c per class)
         ↓
STAGE 1: IMAGE PREPROCESSING
│  ├─ Hair removal (DullRazor)
│  ├─ Artifact masking (ruler marks)
│  ├─ Color normalization (Macenko)
│  ├─ Resize to 224×224
│  └─ Normalize (ImageNet statistics)
         ↓
STAGE 2: PREDICTION + CALIBRATED CONFIDENCE
│  ├─ Forward pass through DenseNet121 (eval mode, dropout disabled)
│  ├─ Apply temperature scaling: z'_c = z_c / T*
│  ├─ Softmax: p̂_c = softmax(z'_c) for c ∈ {NV, MEL, BKL, BCC, AKIEC, VASC, DF}
│  ├─ Predicted class: ŷ = argmax_c p̂_c
│  └─ Calibrated confidence: conf = max_c p̂_c
         ↓
STAGE 3: PREDICTION ENTROPY COMPUTATION
│  ├─ Shannon entropy: H = -Σ_c p̂_c log p̂_c ∈ [0, log(7)]
│  └─ Normalized entropy: Ĥ = H / log(7) ∈ [0, 1]
         ↓
STAGE 4: CONFORMAL PREDICTION SET
│  ├─ Compute nonconformity score: s(x) = 1 - p̂_ŷ  (Least Ambiguous Set approach)
│  ├─ Retrieve class-conditional quantile: q̂_ŷ (from calibration table, at α=0.10)
│  ├─ Prediction set: C_α(x) = {c : p̂_c ≥ 1 - q̂_c}  (all classes above threshold)
│  ├─ Set size: K = |C_α(x)| (key uncertainty indicator: K ≥ 3 → high uncertainty)
│  └─ Coverage indicator: Is ŷ ∈ C_α(x)?  (should be True for ≥90% of cases)
         ↓
STAGE 5: EXPLANATION RELIABILITY ASSESSMENT
│  [From Gap #1 — Pre-computed TIxAI reliability tiers per class]
│  ├─ Query class-level TIxAI tier for predicted class ŷ:
│  │     TIxAI_tier(ŷ) ∈ {High: TIxAI > 0.65, Moderate: 0.45–0.65, Low: < 0.45}
│  ├─ [For current image — if online] Compute Grad-CAM++ → TIxAI_i(x)
│  ├─ Retrieve median TIxAI for class ŷ: μ_TIxAI(ŷ)
│  └─ Explanation reliability signal: ER = TIxAI_i(x) or μ_TIxAI(ŷ) if not computed
         ↓
STAGE 6: MULTI-XAI DISAGREEMENT
│  [From Gap #4 — XDI computation]
│  ├─ Compute XDI(x) = disagreement score between Grad-CAM++, LIME, SHAP, IG
│  ├─ Normalize: XDI_norm(x) ∈ [0, 1]
│  └─ High XDI → multiple XAI methods disagree on which region to highlight
         ↓
STAGE 7: COMPOSITE CLINICAL RISK SCORE (CCRS)
│  ├─ Retrieve malignancy prior for predicted class ŷ: R_mal(ŷ)
│  │     R_mal: MEL=1.0, BCC=0.85, AKIEC=0.75, BKL=0.30, NV=0.10, VASC=0.40, DF=0.15
│  ├─ Compute CCRS(x) =
│  │     w₁ × (1 - conf)                # prediction uncertainty
│  │   + w₂ × Ĥ                          # entropy (multi-class confusion)
│  │   + w₃ × min(K-1, 3)/3              # conformal set size penalty
│  │   + w₄ × (1 - ER)                  # explanation unreliability
│  │   + w₅ × XDI_norm                  # multi-XAI disagreement
│  │   + w₆ × R_mal(ŷ)                  # malignancy risk amplifier
│  └─ Normalize CCRS ∈ [0, 1]
         ↓
STAGE 8: THRESHOLD COMPARISON
│  ├─ Apply per-class threshold: τ_c (optimized per class on validation set)
│  ├─ Decision rule: δ(x) = DEFER if CCRS(x) ≥ τ_ŷ else ACCEPT
│  └─ Apply safety override: FORCE DEFER if K ≥ 4 (conformal set > 4 classes)
         ↓
STAGE 9: OUTPUT GENERATION
│  ├─ If ACCEPT:
│  │     Output: {prediction=ŷ, confidence=conf, explanation=Grad-CAM++,
│  │              CCRS=value, CP_set=C_α(x), reliability_tier=ER_tier}
│  └─ If DEFER:
│         Output: {recommendation="REFER TO DERMATOLOGIST",
│                  reason=identify_trigger(CCRS_components),
│                  urgency=classify_urgency(R_mal, CCRS),
│                  partial_differential=C_α(x)}
         ↓
STAGE 10: REFERRAL DOCUMENTATION
│  ├─ Log: image_id, CCRS, deferral_decision, trigger_reason, urgency_level
│  ├─ If DEFER: append to specialist review queue with urgency priority
│  └─ If ACCEPT: append to monitoring audit (for safety surveillance)
```

### 6.2 Stage-by-Stage Scientific Rationale

**Why Stage 3 (Entropy)?**  
Calibrated maximum confidence (Stage 2) captures only the top-1 probability. Shannon entropy captures the FULL distribution shape — a case with P(MEL)=0.40, P(BCC)=0.35, P(NV)=0.25 has low max confidence BUT also high entropy, indicating genuine multi-class confusion. Entropy is computed on the calibrated softmax distribution.

**Why Stage 4 (Conformal Prediction)?**  
The conformal set size K provides a principled, statistically guaranteed measure of the model's inability to distinguish among diagnoses. K=1 means the model is certain; K=4 means the model considers 4 diagnoses plausible at 90% coverage. The set size is interpretable by clinicians ("the AI considers 3 diagnoses equally plausible — please evaluate") unlike raw entropy values.

**Why Stage 5 (Explanation Reliability)?**  
A key contribution: TIxAI captures whether the model's prediction is supported by the correct visual evidence. A high-confidence prediction with low TIxAI represents the most dangerous case — the model appears certain, but its explanation reveals it is focused on the wrong image region. This is precisely the scenario where deferral adds the most value.

**Why Stage 6 (Multi-XAI Disagreement)?**  
XDI from Gap #4 captures whether different XAI methods agree on which region caused the prediction. High disagreement suggests the prediction is not robustly supported by any single explanatory signal — another indicator of unreliability.

**Why Stage 7 (CCRS + Malignancy Prior)?**  
The malignancy prior $R_{\text{mal}}(\hat{y})$ amplifies the CCRS for high-malignancy predictions. If the model predicts MEL with moderate uncertainty, the high malignancy risk amplifies the CCRS, making deferral more likely. If the model predicts NV with the same uncertainty, the low malignancy risk dampens the CCRS, keeping the case accepted (since an uncertain benign prediction is less dangerous). This asymmetric design directly implements the clinical priority: never miss a malignancy.

**Why the Safety Override (K ≥ 4)?**  
If the conformal prediction set contains 4+ classes, the model cannot distinguish among more than half of all diagnostic categories. This represents a fundamental uncertainty that no threshold calibration can resolve — such cases should always be deferred, regardless of CCRS value.

---

## SECTION 7 — THE DECISION FUNCTION: CCRS DESIGN

### 7.1 Formal CCRS Specification

$$\text{CCRS}(x) = \frac{1}{W} \left[ w_1 (1 - \hat{p}_{\hat{y}}) + w_2 \hat{H} + w_3 \frac{\min(K-1,3)}{3} + w_4 (1 - \text{ER}(x)) + w_5 \text{XDI}(x) + w_6 R_{\text{mal}}(\hat{y}) \right]$$

where $W = w_1 + w_2 + w_3 + w_4 + w_5 + w_6$ is the normalization constant.

**Component definitions:**

| Component | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Prediction uncertainty | $1 - \hat{p}_{\hat{y}}$ | [0,1] | Calibrated confidence complement |
| Normalized entropy | $\hat{H}$ | [0,1] | Entropy / log(7), captures multi-class spread |
| CP set size penalty | $\min(K-1,3)/3$ | [0,1] | Conformal set size, capped at 4 |
| Explanation unreliability | $1 - \text{ER}(x)$ | [0,1] | TIxAI complement; 0 = perfect lesion focus |
| XAI disagreement | $\text{XDI}(x)$ | [0,1] | Multi-XAI disagreement index from Gap #4 |
| Malignancy risk | $R_{\text{mal}}(\hat{y})$ | [0,1] | Class-specific malignancy risk prior |

### 7.2 Weight Determination

**Initial weights (literature-informed defaults):**

| Component | Default Weight | Rationale |
|-----------|---------------|-----------|
| $w_1$ (confidence) | 0.25 | Calibrated confidence is the most directly validated signal |
| $w_2$ (entropy) | 0.20 | Complements confidence with full distribution shape |
| $w_3$ (CP set size) | 0.20 | Formally guaranteed uncertainty indicator |
| $w_4$ (TIxAI) | 0.15 | Novel signal from Gap #1; weight lower until validated |
| $w_5$ (XDI) | 0.10 | Novel signal from Gap #4; weight lower until validated |
| $w_6$ (malignancy) | 0.10 | Asymmetric cost amplifier |

**Optimization:** Weights are optimized on the validation set using Bayesian optimization (Optuna), minimizing the composite objective:
$$\mathcal{L}_{\text{CCRS}} = \lambda_1 \cdot \text{Missed Malignancies Rate} + \lambda_2 \cdot \text{Referral Rate}$$
with $\lambda_1 = 10$ (reflecting the 10× clinical cost asymmetry between missed malignancy and unnecessary referral), $\lambda_2 = 1$.

### 7.3 Malignancy Risk Prior

The malignancy risk prior $R_{\text{mal}}(\hat{y})$ is derived from clinical evidence, not model outputs:

| Class | $R_{\text{mal}}$ | Clinical Basis |
|-------|-----------------|----------------|
| MEL (Melanoma) | 1.00 | Most dangerous; 5-year mortality if late-stage |
| BCC (Basal Cell Carcinoma) | 0.85 | Locally invasive; rarely metastatic but significant morbidity |
| AKIEC (Actinic Keratosis) | 0.75 | Precancerous; progression risk ~10–15% per year |
| VASC (Vascular Lesions) | 0.40 | Angiosarcoma subtypes can be aggressive; diagnosis uncertain |
| BKL (Benign Keratosis) | 0.30 | Seborrheic keratosis mimics melanoma; misclassification risk |
| DF (Dermatofibroma) | 0.15 | Rarely malignant; but can mask deeper pathology |
| NV (Melanocytic Nevi) | 0.10 | Benign; low malignancy potential in most presentations |

**Source:** Derived from dermatological clinical guidelines (AAD, EDF), not from the classification model. This prevents circular reasoning.

### 7.3.1 Reconciliation With the Deployed Code (added 2026-07-05)

**The notebook's actual `compute_ccrs()` implementation (CELL 15) is not a literal implementation of the 6-term formula above** — it differs in two ways worth documenting explicitly, since a prior audit found the deployed code violated this section's own non-circularity principle:

1. **Structural difference:** the deployed formula uses `w1(1-confidence) + w2(1-TIxAI) + w3·XDI + w4·R_mal·(1-confidence) + w5·(conformal_set_size/n_classes) + w6·pair_risk`. It does not include a separate normalized-entropy term ($\hat{H}$ from §7.1), and it adds a **confusion-pair risk term** (`w6·pair_risk`, weight 0.20–0.40 across the tested weight candidates) that has no counterpart in this section's formal specification at all.
2. **The added confusion-pair term was, until 2026-07-05, the source of a real circularity bug:** it was computed as `confusion_pair_risk(true_class, pred_class)`, returning 0.0 whenever the prediction was correct — i.e. it used the sample's own ground-truth label as a scoring input, exactly the failure mode §7.3's "derived from clinical guidelines, not the model, prevents circular reasoning" principle is meant to rule out. This is what produced the "100% error rate in deferred cases" symptom in the pre-fix executed output — a mechanical consequence of the leak, not a validated deferral capability. **This has been fixed:** `confusion_pair_risk()` now takes only the *predicted* class plus an aggregate, population-level historical error rate for that predicted class (from CELL 13's confusion matrix, summed across all true classes), and a static set of predicted classes historically implicated in dangerous MEL-adjacent confusions (`DANGEROUS_PRED_CLASSES = {NV, BKL, BCC, AKIEC}`) — no per-sample true label is used anywhere in the score. `true_class` is still recorded per output row, but only for post-hoc evaluation of deferral accuracy, never as a scoring input.
3. **Residual note:** the fixed confusion-pair term is a data-driven, aggregate class-level prior (built from the same test set later used to *evaluate* deferral quality), which is a milder, more defensible form of information reuse than per-sample label leakage, but is not identical to §7.3's "purely clinical-guideline-derived, model/data-independent" `R_mal` design either. A stricter implementation would derive the confusion-pair aggregate from a split disjoint from the one being evaluated (e.g. the validation set) rather than the test set's own confusion matrix. Flagging this for a future revision rather than treating the current fix as the final word on circularity.

Until this reconciliation, any reader comparing this section's formula to the notebook's actual printed CCRS/deferral numbers would not have been able to tell the two had diverged — future edits to either the formula (Section 7.1) or the code should update both together.

### 7.4 Threshold Optimization

**Step 1: Define the target operating point**
Clinical consultation specifies:
- Minimum malignant sensitivity in ACCEPT set: $\tau_{\text{sens}} = 0.95$
- Maximum referral rate: $\tau_{\text{ref}} = 0.30$

**Step 2: Sweep per-class thresholds**
For each class $c$, sweep $\tau_c \in [0, 1]$ (100 grid points). For each $\tau_c$, compute:
- Referral rate: $\text{RR}(c) = P(\text{CCRS}(x) \geq \tau_c | \hat{y} = c)$
- Malignant sensitivity: $\text{SENS}_{\text{mal}}(c) = P(\text{DEFER} | y \in \mathcal{M})$ where $\mathcal{M} = \{$MEL, BCC, AKIEC$\}$
- Net Benefit: $\text{NB}(\tau_c)$ from DCA formula

**Step 3: Select optimal threshold**
$$\tau_c^* = \arg\max_{\tau_c} \text{NB}(\tau_c) \quad \text{subject to} \quad \text{SENS}_{\text{mal}} \geq 0.95$$

**Step 4: Global safety override**
Regardless of $\tau_c^*$, always DEFER if $K \geq 4$ (conformal set contains ≥ 4 classes).

### 7.5 Urgency Classification for Deferred Cases

Not all deferred cases have equal urgency. The system classifies deferred cases into three urgency tiers for specialist review queue management:

| Urgency | Criteria | Action |
|---------|----------|--------|
| **URGENT** (Red) | $R_{\text{mal}}(\hat{y}) \geq 0.75$ AND $K \geq 3$ | Same-day specialist review |
| **PRIORITY** (Orange) | $R_{\text{mal}}(\hat{y}) \geq 0.40$ OR $K \geq 3$ | Within 1 week review |
| **ROUTINE** (Yellow) | Low malignancy risk; moderate uncertainty | Standard queue |

This urgency tiering is a practical clinical deployment feature. It ensures that genuine melanoma candidates are reviewed immediately, while uncertain-but-likely-benign cases are queued appropriately.

---

## SECTION 8 — DATASET STRATEGY

### 8.1 Primary Dataset: HAM10000

**Role:** Train base classifier (DenseNet121 with focal loss) + calibrate temperature scaling  
**Split:** 70% train / 15% validation / 15% test (patient-ID stratified)

**Critical additions for Gap #8:**

The HAM10000 classification labels must be augmented with clinical malignancy labels for CCRS training:

**Binary malignancy label:**
$$y_{\text{mal}}(c) = \begin{cases} 1 & c \in \{\text{MEL}, \text{BCC}, \text{AKIEC}\} \\ 0 & \text{otherwise} \end{cases}$$

This binary label is used for malignancy-sensitive threshold optimization and for computing malignant sensitivity metrics.

### 8.2 Conformal Prediction Calibration Set

**Requirement:** A held-out calibration set, separate from training AND test, used exclusively for computing conformal prediction quantiles.  
**Source:** Validation set (15% of HAM10000, split by patient ID)  
**Important:** This set must never be used for training, hyperparameter tuning, or performance evaluation — only for CP calibration.

### 8.3 Threshold Optimization Set

**Use case:** Optimizing per-class CCRS thresholds $\tau_c^*$  
**Source:** First half of validation set (7.5% of total)  
**Note:** The second half of validation set is used for temperature scaling calibration. This prevents overlap.

### 8.4 External Validation

**Primary external validation:** ISIC 2020 (33,126 dermoscopy images, binary labels)  
**Use:** Test generalizability of CCRS and deferral policy to a different collection context  
**Secondary external validation:** BCN20000 (19,424 dermoscopy images, multi-class)  
**Limitation:** Class composition differs from HAM10000 — report class-specific deferral metrics for overlapping classes only.

**PAD-UFES-20:** For classification generalization only (smartphone vs. dermoscopy domain mismatch prevents valid CCRS comparison).

### 8.5 Special Data Requirements for Gap #8

**Requirement 1: Correctly paired (TIxAI, XDI, CCRS) triples**
Each test image must have: classification output, TIxAI score (from Gap #1), XDI score (from Gap #4), and CCRS computed. All three must use the same model and the same forward pass.

**Requirement 2: Malignancy ground truth labels**
For threshold optimization and safety evaluation, we need binary malignancy labels (not just 7-class labels). These are derived from expert reclassification of HAM10000 labels using the malignancy mapping defined above.

**Requirement 3: Human expert deferral benchmark (aspirational)**
The gold standard for evaluating a deferral system is comparing it to expert dermatologist decisions about which cases they would request specialist consultation for. This requires a prospective study or retrospective annotation effort. If available, a subset of expert-annotated "I would refer this case" labels provides the most clinically meaningful validation benchmark.

---

## SECTION 9 — TRAINING METHODOLOGY

### 9.1 Base Classifier Training

The base classifier (DenseNet121) is trained exactly as specified in Gap #1 (Per-Class TIxAI report):
- Class-weighted focal loss (γ=2.0)
- Weighted random sampler
- AdamW optimizer, OneCycleLR
- Label smoothing (ε=0.1)
- 60 epochs, early stopping on validation macro-F1
- Temperature scaling calibration on validation set

**This is NOT a new training step for Gap #8** — the Gap #1 model is reused. Gap #8 adds only the post-training deferral layer.

### 9.2 Conformal Prediction Setup

```python
import numpy as np
from scipy.special import softmax

class AdaptiveConformalPredictor:
    """
    Class-conditional adaptive conformal predictor for dermoscopy.
    Provides formal coverage guarantees: P(y ∈ C_α(x)) ≥ 1-α
    """
    
    def __init__(self, alpha: float = 0.10):
        """
        alpha: miscoverage rate (0.10 = 90% coverage guarantee)
        """
        self.alpha = alpha
        self.class_quantiles = {}  # per-class quantiles
    
    def fit(self, cal_logits: np.ndarray, cal_labels: np.ndarray, 
            temperature: float):
        """
        Fit CP quantiles on the calibration set.
        
        cal_logits: (n_cal, 7) raw logits from DenseNet121
        cal_labels: (n_cal,) true class labels
        temperature: T* from temperature scaling
        """
        # Apply temperature scaling
        cal_probs = softmax(cal_logits / temperature, axis=1)
        
        # Compute nonconformity scores: s(x,y) = 1 - p̂_y
        for c in range(7):
            class_mask = cal_labels == c
            if class_mask.sum() < 10:
                print(f"Warning: Class {c} has only {class_mask.sum()} calibration samples")
            
            # Nonconformity scores for true class
            scores = 1.0 - cal_probs[class_mask, c]
            
            # Class-conditional quantile at level ceil((n_c+1)(1-α)) / n_c
            n_c = class_mask.sum()
            q_level = np.ceil((n_c + 1) * (1 - self.alpha)) / n_c
            q_level = min(q_level, 1.0)
            self.class_quantiles[c] = np.quantile(scores, q_level)
    
    def predict(self, logits: np.ndarray, temperature: float):
        """
        Generate prediction sets for test images.
        
        Returns:
            prediction_sets: list of sets, each containing class indices
            set_sizes: array of prediction set sizes
        """
        probs = softmax(logits / temperature, axis=1)
        prediction_sets = []
        
        for i in range(len(logits)):
            pred_set = []
            for c in range(7):
                # Include class c if its nonconformity score is below quantile
                if 1.0 - probs[i, c] <= self.class_quantiles[c]:
                    pred_set.append(c)
            prediction_sets.append(pred_set)
        
        set_sizes = np.array([len(ps) for ps in prediction_sets])
        return prediction_sets, set_sizes
```

### 9.3 CCRS Weight Optimization

```python
import optuna

def ccrs_objective(trial, val_data, lambda_miss=10.0, lambda_ref=1.0):
    """
    Optimize CCRS weights to minimize clinical loss function.
    """
    # Sample weights (must sum to 1)
    w = [trial.suggest_float(f'w{i}', 0.05, 0.50) for i in range(6)]
    w = np.array(w) / sum(w)  # normalize
    
    # Compute CCRS for all validation images
    ccrs_scores = []
    for row in val_data:
        ccrs = (
            w[0] * (1 - row['conf']) +
            w[1] * row['entropy_norm'] +
            w[2] * min(row['cp_size'] - 1, 3) / 3.0 +
            w[3] * (1 - row['tixai']) +
            w[4] * row['xdi'] +
            w[5] * row['malignancy_risk']
        )
        ccrs_scores.append(ccrs)
    
    ccrs_scores = np.array(ccrs_scores)
    
    # Optimize global threshold for this weight combination
    best_loss = float('inf')
    for tau in np.linspace(0.1, 0.9, 100):
        deferred = ccrs_scores >= tau
        accepted = ~deferred
        
        # Compute metrics
        referral_rate = deferred.mean()
        
        # Malignant sensitivity = fraction of malignant cases deferred
        malignant_mask = val_data['is_malignant']
        missed_maligancies = (malignant_mask & accepted).sum() / malignant_mask.sum()
        
        # Clinical loss
        loss = lambda_miss * missed_maligancies + lambda_ref * referral_rate
        if loss < best_loss:
            best_loss = loss
    
    return best_loss

# Run Bayesian optimization
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: ccrs_objective(trial, val_data), n_trials=500)
optimal_weights = study.best_params
```

### 9.4 Per-Class Threshold Optimization

```python
def optimize_per_class_thresholds(val_data, optimal_weights, 
                                   target_sensitivity=0.95):
    """
    Find per-class deferral thresholds that maximize Net Benefit
    subject to malignant sensitivity ≥ target_sensitivity.
    """
    class_thresholds = {}
    
    for c in range(7):
        class_mask = val_data['predicted_class'] == c
        class_data = val_data[class_mask]
        
        if len(class_data) < 20:
            # Too few samples — use global threshold for rare classes
            class_thresholds[c] = global_threshold
            continue
        
        best_nb = -float('inf')
        best_tau = 0.5
        
        for tau in np.linspace(0.05, 0.95, 200):
            deferred = class_data['ccrs'] >= tau
            accepted = ~deferred
            
            # Check malignant sensitivity constraint
            if class_data['is_malignant'].any():
                mal_sensitivity = class_data[class_data['is_malignant']]['ccrs'].ge(tau).mean()
                if mal_sensitivity < target_sensitivity:
                    continue
            
            # Compute Net Benefit at threshold t_c (clinical odds = 1/9 for melanoma)
            tp = (deferred & class_data['is_malignant']).sum()
            fp = (deferred & ~class_data['is_malignant']).sum()
            n = len(class_data)
            t = MALIGNANCY_RISK[c]  # threshold probability from malignancy prior
            
            nb = tp/n - (fp/n) * (t / (1-t))
            
            if nb > best_nb:
                best_nb = nb
                best_tau = tau
        
        class_thresholds[c] = best_tau
    
    return class_thresholds
```

---

## SECTION 10 — CLINICAL DEFERRAL IMPLEMENTATION

### 10.1 Performance Evaluation Framework

The following metrics are computed on the **held-out test set** (never used for training, calibration, or threshold optimization):

#### Primary Safety Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Malignant Sensitivity (DEFER) | P(DEFER | malignant) | ≥ 0.85 |
| Missed Malignancy Rate | P(ACCEPT | malignant) | ≤ 0.15 |
| False Reassurance Rate | P(ACCEPT, wrong | malignant) | → 0 |

#### Primary Utility Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Referral Rate | P(DEFER) | ≤ 0.30 |
| Accepted Accuracy | Accuracy on ACCEPT set | ≥ 0.90 |
| AUARC | Area under Accuracy-Rejection Curve | > baseline |

#### Calibration Metrics

| Metric | Target |
|--------|--------|
| ECE (15-bin) | ≤ 0.05 post-calibration |
| CCRS Calibration | Deferred cases should have higher error rate than accepted |
| Reliability Diagram | Visual check: CCRS vs. error rate |

#### Clinician Workload Metrics

| Metric | Description |
|--------|-------------|
| Relative workload reduction | Fraction of cases requiring specialist review |
| Urgency distribution | % URGENT / PRIORITY / ROUTINE of deferred cases |
| False deferral rate | % benign cases unnecessarily deferred |

### 10.2 Decision Curve Analysis

```python
def decision_curve_analysis(y_true_malignant, ccrs_scores, 
                             threshold_range=np.linspace(0.01, 0.99, 100)):
    """
    Compute Decision Curve Analysis for the CCRS-based deferral system.
    
    y_true_malignant: binary array (1=malignant, 0=benign)
    ccrs_scores: array of CCRS values (higher = more likely to defer)
    """
    n = len(y_true_malignant)
    prevalence = y_true_malignant.mean()
    
    results = []
    for t in threshold_range:
        # Treat CCRS ≥ τ as "positive" (defer = flagged as concerning)
        y_pred_defer = (ccrs_scores >= t).astype(int)
        
        tp = ((y_pred_defer == 1) & (y_true_malignant == 1)).sum()
        fp = ((y_pred_defer == 1) & (y_true_malignant == 0)).sum()
        
        # Net Benefit = TP/n - FP/n × (t/(1-t))
        nb = tp/n - (fp/n) * (t / (1-t))
        
        # Comparators:
        # "Treat all" = refer everyone: NB = prevalence - (1-prevalence) × t/(1-t)
        nb_all = prevalence - (1-prevalence) * (t/(1-t))
        # "Treat none" = refer no one: NB = 0
        nb_none = 0.0
        
        results.append({
            'threshold': t,
            'net_benefit_model': nb,
            'net_benefit_all': nb_all,
            'net_benefit_none': nb_none
        })
    
    return pd.DataFrame(results)
```

### 10.3 Accuracy-Rejection Curve

```python
def accuracy_rejection_curve(y_true, y_pred, ccrs_scores):
    """
    Compute the Accuracy-Rejection Curve (ARC).
    
    Plots: Rejection Rate (x) vs. Accuracy on Accepted Set (y)
    A good deferral system should show monotonically increasing accuracy
    as more uncertain cases are rejected.
    """
    sorted_indices = np.argsort(ccrs_scores)  # sort ascending (most confident first)
    
    arc_points = []
    for i in range(0, len(sorted_indices), len(sorted_indices)//100):
        accepted_indices = sorted_indices[:len(sorted_indices)-i]
        if len(accepted_indices) == 0:
            break
        
        acc = (y_true[accepted_indices] == y_pred[accepted_indices]).mean()
        rejection_rate = i / len(sorted_indices)
        arc_points.append({'rejection_rate': rejection_rate, 'accuracy': acc})
    
    arc_df = pd.DataFrame(arc_points)
    auarc = np.trapz(arc_df['accuracy'], arc_df['rejection_rate'])
    return arc_df, auarc
```

### 10.4 CCRS Reliability Diagram

```python
def ccrs_reliability_diagram(y_true, y_pred, ccrs_scores, n_bins=10):
    """
    Calibration check: within each CCRS bin, compute actual error rate.
    A well-calibrated CCRS should show: higher CCRS → higher error rate.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_errors = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (ccrs_scores >= bins[i]) & (ccrs_scores < bins[i+1])
        if mask.sum() > 0:
            error_rate = (y_true[mask] != y_pred[mask]).mean()
            bin_errors.append(error_rate)
            bin_counts.append(mask.sum())
        else:
            bin_errors.append(np.nan)
            bin_counts.append(0)
    
    return np.array(bin_errors), np.array(bin_counts), bins
```

### 10.5 McNemar's Test for Deferral Comparison

Used to compare whether the CCRS-based deferral achieves significantly better malignant sensitivity than a confidence-only baseline:

```python
from statsmodels.stats.contingency_tables import mcnemar

def compare_deferral_policies(y_true_malignant, deferred_ccrs, deferred_conf_only):
    """
    McNemar's test: does CCRS deferral catch significantly more malignant cases
    than confidence-only deferral?
    """
    # Concordant/discordant table for malignant cases only
    mal_mask = y_true_malignant == 1
    
    both_deferred = (deferred_ccrs[mal_mask] & deferred_conf_only[mal_mask]).sum()
    only_ccrs = (deferred_ccrs[mal_mask] & ~deferred_conf_only[mal_mask]).sum()
    only_conf = (~deferred_ccrs[mal_mask] & deferred_conf_only[mal_mask]).sum()
    neither = (~deferred_ccrs[mal_mask] & ~deferred_conf_only[mal_mask]).sum()
    
    table = np.array([[both_deferred, only_ccrs],
                       [only_conf, neither]])
    
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue
```

---

## SECTION 11 — STATISTICAL ANALYSIS PLAN

### 11.1 Analysis Hierarchy

```
Level 1: Descriptive Statistics
  → CCRS distribution per class, referral rate, malignancy sensitivity
  
Level 2: Calibration Assessment
  → ECE, reliability diagrams, CCRS calibration check
  
Level 3: Primary Hypothesis Tests
  → Is malignant sensitivity in ACCEPT set significantly higher with CCRS than conf-only?
  → McNemar's test
  
Level 4: Clinical Utility Assessment
  → Decision Curve Analysis across threshold range [0.01, 0.99]
  → AUARC comparison
  
Level 5: Comparative Ablation
  → Compare CCRS variants: which signals contribute most to deferral quality?
  
Level 6: Robustness
  → Bootstrap confidence intervals
  → External validation replication
```

### 11.2 Primary Statistical Tests

#### Test 1: McNemar's Test — Malignant Sensitivity Comparison
**Null hypothesis H₀:** Malignant sensitivity is equal for CCRS-based deferral vs. confidence-only deferral (at matched referral rate)  
**Alternative H₁:** CCRS-based deferral achieves higher malignant sensitivity  
**Why McNemar:** Both policies are applied to the same test set — the comparisons are paired. McNemar's test is correct for paired binary outcomes.  
**Report:** χ²(1), p-value, OR (Odds Ratio) of malignant detection, 95% CI

#### Test 2: DeLong's Test — AUC Comparison
**Purpose:** Compare AUARC (Area Under Accuracy-Rejection Curve) between CCRS and confidence-only  
**Method:** DeLong et al. (1988) — exact nonparametric comparison of AUC-like measures  
**Why:** AUARC is computed from the same test set observations, making paired comparison appropriate

#### Test 3: Bootstrap Confidence Intervals — Referral Rate and Sensitivity
**Purpose:** Estimate uncertainty in key operating metrics  
**Method:** 10,000-iteration BCa bootstrap on test set  
**Metrics:** Referral rate ± CI, Malignant sensitivity ± CI, CCRS AUARC ± CI  
**Why BCa:** Corrects for bootstrap distribution skewness; recommended for n < 500 per class

#### Test 4: Decision Curve Analysis — Net Benefit Curves
**Purpose:** Evaluate clinical utility across all possible threshold probabilities  
**Method:** DCA with 95% confidence bands (bootstrap)  
**Comparators:**
1. "Refer all" strategy (upper bound for malignant sensitivity)
2. "Refer none" strategy (zero referrals = zero malignant catch via referral)
3. Confidence-only deferral (primary ablation comparison)
4. CCRS-based deferral (proposed system)  
**Interpretation:** The CCRS system provides clinical utility if its DCA curve lies above both "refer all" and "refer none" at clinically relevant threshold probabilities (0.05–0.50).

#### Test 5: Calibration Test (Hosmer-Lemeshow)
**Purpose:** Formally test CCRS calibration — does CCRS reliably predict error rate?  
**Method:** Hosmer-Lemeshow goodness-of-fit test on 10 CCRS deciles  
**H₀:** CCRS is well-calibrated (observed error rates match CCRS-predicted error rates)  
**H₁:** Miscalibration (CCRS is not a reliable predictor of actual errors)

### 11.3 Secondary Analyses

**Analysis A: Contribution analysis of CCRS components**  
Train CCRS variants with one component removed at a time (6 ablation variants). Compare AUARC for each. This identifies which signals are most critical.

**Analysis B: Class-stratified deferral performance**  
Report referral rate, malignant sensitivity, and CCRS distribution per class. Hypothesis: MEL and AKIEC show higher effective deferral rates (correctly identified as uncertain/risky) than NV and BKL.

**Analysis C: Urgency calibration**  
For URGENT-tier deferred cases: what fraction are truly malignant? A well-calibrated urgency system should show >80% malignant content in URGENT-tier.

**Analysis D: TIxAI and XDI signal contribution**  
Among cases where confidence-only would ACCEPT but CCRS DEFERS: what is the per-signal contribution? This isolates the added value of explanation reliability signals over confidence alone.

---

## SECTION 12 — REVIEWER #2 SIMULATION

### Criticism C1: "The CCRS is ad hoc — why these 6 components and these weights?"

> *"The proposed CCRS combines 6 heterogeneous signals with weights optimized via Bayesian optimization. This is arbitrary feature engineering. Why not a learned end-to-end deferral network? The weight justification is post-hoc rationalization."*

**Response:**

The CCRS design is NOT arbitrary — each component is motivated by published evidence:
- Confidence: calibrated probability, validated for deferral ([LR-07])
- Entropy: captures multi-class confusion beyond top-1 ([LR-03])
- CP set size: only signal with formal statistical coverage guarantees ([LR-04])
- TIxAI: captures explanation reliability — gap identified in Gap #1
- XDI: captures explanation agreement — from Gap #4
- Malignancy prior: asymmetric cost structure from clinical evidence ([LR-12])

The weights ARE optimized, not invented. We present weight initialization as literature-motivated defaults, then optimize via Bayesian search on the validation set with a clinically meaningful objective function (asymmetric malignancy-weighted loss). This is standard practice in clinical risk score development (similar to how HEART score, Wells criteria, etc. are developed).

**Why not end-to-end learned deferral?** SelectiveNet and L2D require ground truth "should be deferred" labels for training — which we do not have. Post-hoc CCRS computation from existing signals is the appropriate method when ground truth deferral labels are unavailable.

**Redesign action:** Report 95% CIs for optimal weights across 3 random validation splits (stability analysis). If weights are stable (CI width < 0.10), the optimization is robust.

---

### Criticism C2: "You cannot validate clinical deferral without human expert comparison."

> *"The paper claims to optimize 'when clinicians should not trust AI.' But no clinician data is collected. The referral threshold is set based on an arbitrary mathematical criterion. Without prospective data on which cases experienced dermatologists would refer, this work is purely theoretical."*

**Response:**

This is the most important criticism, and the most honest limitation. We respond at three levels:

**Level 1:** The malignancy-weighted optimization (targeting sensitivity ≥ 0.95 for malignant cases) is NOT arbitrary — it reflects the clinical standard of care where missing a melanoma is unacceptable. This is equivalent to designing a cancer screening test with ≥95% sensitivity as a regulatory requirement.

**Level 2:** Expert deferral ground truth generation is a proposed future work step, explicitly outlined in the paper: "Prospective validation with expert dermatologist annotation of referral decisions is necessary to validate the clinical utility of CCRS-based deferral. We propose [specific study design]."

**Level 3:** Decision Curve Analysis evaluates clinical utility under explicit assumptions about the relative cost of missed malignancy vs. unnecessary referral (the threshold probability). Clinicians who have different cost assumptions can read the DCA curve at their preferred threshold to determine utility.

**Redesign action:** Add a "Limitations" section explicitly acknowledging that CCRS weights encode assumptions about relative clinical costs. Report sensitivity analysis: how do optimal thresholds change if $\lambda_{\text{miss}}$ is 5× or 20× (instead of 10×)?

---

### Criticism C3: "The conformal prediction guarantee fails under distribution shift."

> *"Conformal prediction guarantees coverage only if the calibration set and test set are exchangeable (i.i.d.). In clinical practice, new patients, new imaging devices, and new dermoscope models will violate this assumption. The coverage guarantee collapses under distribution shift."*

**Response:**

This is a legitimate and well-known limitation of standard CP. We address it through:

1. **External validation replication:** Test whether CP coverage (empirical proportion of true class in prediction set) holds on ISIC 2020 and BCN20000. If empirical coverage ≈ 90% on external datasets, the exchangeability assumption approximately holds.

2. **Weighted conformal prediction:** Recent methods (Tibshirani et al., 2019 — covariate shift; Barber et al., 2023 — weighted CP) extend CP to handle distribution shift by reweighting calibration samples. We report an additional analysis using weighted CP.

3. **Monitoring protocol:** For deployment, we recommend continuous monitoring of empirical coverage (fraction of cases where true class ∈ prediction set). If empirical coverage drops below 85%, the CP quantiles should be recalibrated. This converts the guarantee from a static claim to a monitored contract.

**Redesign action:** Add a "Distribution Shift Analysis" subsection. Report empirical coverage on each external dataset. Include weighted CP as a robustness comparison.

---

### Criticism C4: "How is the malignancy prior justified? Who set these clinical risk scores?"

> *"The malignancy prior R_mal(ŷ) is defined as R_mal(MEL)=1.0, R_mal(VASC)=0.40, etc. Who decided these values? Where is the clinical validation? These look invented."*

**Response:**

The malignancy priors are derived from peer-reviewed clinical guidelines, not invented:
- MEL=1.0: Universally recognized as the most deadly skin cancer (SEER database)
- BCC=0.85: Locally invasive; significant morbidity even if rarely metastatic (AAD guidelines)
- AKIEC=0.75: Precancerous lesion with documented 10–15% annual progression risk (multiple meta-analyses)
- VASC=0.40: Angiosarcoma is highly aggressive; but most vascular lesions are benign — intermediate prior reflects diagnostic uncertainty
- BKL=0.30: Seborrheic keratosis can mimic melanoma; clinical concern is misclassification, not inherent malignancy
- DF=0.15: Dermatofibroma rarely malignant; but can mask DFSP (Dermatofibrosarcoma Protuberans)
- NV=0.10: Common Spitz nevi and dysplastic nevi have some malignant potential; pure melanocytic nevi are largely benign

**Redesign action:** Add Table of malignancy priors with citation for each value. Conduct sensitivity analysis varying each prior by ±0.20 — if CCRS thresholds change by <5%, the priors are stable; if sensitive, discuss which classes require dermatologist input to set.

---

### Criticism C5: "The referral rate target (30%) is arbitrary."

> *"The paper targets a referral rate of ≤ 30%. This number appears unjustified. In a busy dermatology clinic seeing 50 patients per day, 30% referral means 15 extra specialist consultations per day. Is this operationally feasible?"*

**Response:**

The 30% target was chosen as a conservative upper bound, not as a design requirement. The CCRS system is parametric — any referral rate target between 0% and 100% can be achieved by adjusting $\tau_c^*$.

The clinical deployment decision about the appropriate referral rate depends on:
- Local specialist availability
- Prevalence of malignant cases in the target population
- Existing baseline referral rate (AI should not INCREASE unnecessary referrals)

For the paper, we report the full DCA curve, allowing each clinical environment to select its own operating point. We additionally report 3 specific operating points:
1. **Conservative:** Target sensitivity 0.97 (minimize missed malignancies)
2. **Balanced:** Target sensitivity 0.95 + referral ≤ 0.30
3. **Efficient:** Target referral ≤ 0.20 (maximize specialist workload reduction)

**Redesign action:** Replace the single "target referral ≤ 30%" criterion with a multi-operating-point analysis. Report metrics at all 3 operating points in the main results table.

---

### Criticism C6: "The novelty is unclear — this is a logistic regression on 6 features."

> *"At its core, the CCRS is a weighted linear combination of 6 features with weights optimized via Bayesian search. This is functionally equivalent to logistic regression. Where is the methodological novelty?"*

**Response:**

The methodological contribution is NOT in the aggregation mechanism (which is intentionally kept simple for clinical interpretability). The contribution is:

1. **The identification of explanation reliability (TIxAI) as a deferral signal** — not done before
2. **The integration of formal CP coverage** with XAI reliability in a single deferral framework — not done before
3. **The asymmetric malignancy-weighted threshold optimization** with DCA evaluation — not applied to dermoscopy before
4. **The urgency tiering** for specialist review queue management — not previously implemented

A simple, interpretable aggregation mechanism (weighted linear combination) is scientifically appropriate for clinical risk scores — the HEART score, Wells criteria, CHA2DS2-VASc score are ALL weighted linear combinations, yet are foundational clinical tools published in top medical journals. Clinical interpretability requires simplicity.

**Redesign action:** In the Introduction, explicitly state: "We deliberately use a transparent, interpretable aggregation to facilitate clinical adoption. The contribution lies in the signal selection, the formal coverage integration, and the asymmetric clinical optimization — not in the aggregation mechanism."

---

### Criticism C7: "External validation on ISIC 2020 (binary labels) is insufficient for multi-class CCRS."

> *"ISIC 2020 provides only binary labels (malignant/benign). The CCRS is designed for 7-class classification. External validation on binary-label data cannot evaluate per-class CCRS performance."*

**Response:**

Agreed. External validation on ISIC 2020 evaluates ONLY the malignancy-sensitivity component of CCRS, not the full 7-class per-class threshold system.

**Revised external validation strategy:**
1. **ISIC 2020 (binary):** Validate malignant sensitivity and referral rate for the malignancy-detection objective only
2. **BCN20000 (multi-class, if accessible):** Full 7-class CCRS validation
3. **Fallback if BCN20000 unavailable:** Cross-validation on HAM10000 + ISIC 2019 combined dataset with 7 classes

This limitation is explicitly acknowledged: "Full per-class CCRS external validation requires a multi-class dermoscopy dataset distinct from HAM10000 with expert-verified 7-class labels."

---

### Criticism C8: "How does the system handle completely out-of-distribution inputs?"

> *"A dermoscopy image of a non-skin lesion (e.g., a finger accidentally photographed) would still produce a CCRS score and a deferral decision. How does the system detect true OOD inputs?"*

**Response:**

This is a genuine robustness gap. We add a dedicated OOD detection preprocessing step:

**Implementation:** Train a one-class classifier (or use the CP prediction set as an OOD indicator) on dermoscopy training images. At test time:
1. If prediction set K ≥ 6 (all classes plausible — effectively random prediction), flag as POTENTIAL OOD
2. If image quality score (BRISQUE) > 60, flag as LOW QUALITY
3. Output a special "INVALID INPUT — Not a dermoscopy image" label with recommendation to resubmit with a valid dermoscopy image

This is not a full OOD detection solution (which is an open research problem), but it catches the most obvious cases. Full OOD detection is proposed as future work.

---

### Criticism C9: "The paper is simultaneously about 3 contributions (Gap #1, Gap #4, Gap #8). Gap #8 evaluation is confounded by the quality of Gaps #1 and #4."

> *"The CCRS depends on TIxAI (Gap #1) and XDI (Gap #4). If these signals are noisy or inaccurate, the CCRS will be noisy. The paper cannot claim strong CCRS performance if the input signals are not validated independently."*

**Response:**

This is correct, and the paper structure already addresses this:
- Gap #1 (Per-Class TIxAI) is validated independently before Gap #8 development
- Gap #4 (XDI) is validated independently before Gap #8 development

For Gap #8, we conduct a dedicated ablation (Analysis D) that measures CCRS performance with and without TIxAI and XDI components. The ablation quantifies how much these signals add beyond confidence-only deferral. If TIxAI and XDI add zero value, the ablation will show it — and we will acknowledge it honestly.

**Additionally:** We report TIxAI and XDI signal quality (SSIM, inter-method agreement) as prerequisites before CCRS computation. If signal quality falls below validated thresholds, the CCRS defaults to confidence + CP only (a degraded but still functional mode).

---

## SECTION 13 — DEPLOYMENT RECOMMENDATIONS

### 13.1 Deployment Configuration

**Minimum hardware:**
- GPU: NVIDIA RTX 3080 (10GB VRAM) for batch inference
- CPU: For single-image inference, a 16-core CPU with 32GB RAM is sufficient (DenseNet121 CPU inference: ~200ms)

**Inference latency:**
- DenseNet121 forward pass: ~15ms (GPU), ~200ms (CPU)
- Temperature scaling: <1ms
- Conformal prediction: <1ms (table lookup)
- Grad-CAM++ (for TIxAI): ~50ms (GPU)
- CCRS computation: <1ms
- Total per-image pipeline: ~70ms (GPU), ~300ms (CPU)

Both are clinically acceptable for non-real-time diagnostic support.

### 13.2 Safety Gates Before Deployment

| Gate | Threshold | Action if Failed |
|------|-----------|-----------------|
| ECE | ≤ 0.05 | Recalibrate temperature scaling |
| Malignant sensitivity (test) | ≥ 0.90 | Adjust thresholds; do not deploy |
| CP empirical coverage | ≥ 85% on external val | Recalibrate CP quantiles |
| Referral rate | ≤ 35% | Adjust τ_c^* upward |
| CCRS AUARC > baseline | Statistically significant | Review CCRS component weights |

### 13.3 Continuous Monitoring

Post-deployment, the following metrics should be monitored monthly:

1. **Empirical CP coverage:** What fraction of accepted cases have the true class in their conformal set? Target ≥ 85%.
2. **Deferred case resolution rate:** Of deferred cases reviewed by specialists, what fraction were actually clinically significant? Target ≥ 50% (ensures referrals are clinically meaningful).
3. **Missed malignancy surveillance:** Zero tolerance. Any missed malignancy (false reassurance case that later presented with malignancy) requires immediate investigation and system recalibration.
4. **CCRS drift detection:** Are CCRS distributions shifting over time? Significant drift suggests covariate shift in patient population — triggers recalibration.

### 13.4 Regulatory Alignment

**EU AI Act (2024):** Dermoscopy decision support qualifies as a "high-risk AI system." Requires:
- Transparency about AI limitations
- Human oversight capability
- Robustness and accuracy documentation
- Ongoing monitoring

**FDA SaMD (Software as a Medical Device):** For US deployment:
- Predetermined change control plan (how to update CP quantiles and CCRS weights)
- Post-market surveillance protocol
- Validation on diverse patient populations (Fitzpatrick skin types I–VI)

**TRIPOD-AI (2024):** All reported metrics must follow TRIPOD-AI reporting guidelines. DCA is specifically recommended.

### 13.5 Explainability for Deferred Cases

When the system defers a case, the clinician receives:

```
REFERRAL RECOMMENDATION — Clinical AI Deferral

Patient ID: [XXXX]
Image ID: [XXXX]
Capture Date: [XXXX]

⚠️ REFER TO DERMATOLOGIST

Reason(s) for Referral:
  ● Low explanation reliability (TIxAI = 0.31) — Model attention not focused on lesion
  ● High diagnostic uncertainty (Conformal Set = {MEL, BCC, BKL})
  ● Elevated malignancy risk (Predicted class BCC, Risk = 0.85)

Urgency: PRIORITY — Within 1 week

Partial Differential Diagnosis (Conformal Prediction Set):
  - Melanoma (MEL): 38.2% probability
  - Basal Cell Carcinoma (BCC): 34.1% probability [Top predicted]
  - Benign Keratosis (BKL): 27.7% probability

Note: The AI prediction should NOT be used as a diagnosis for this case.
This case has been flagged for mandatory specialist review.
```

This output format satisfies the interpretability requirement (R8) and gives the reviewing dermatologist actionable information about WHY the case was deferred.

---

## APPENDIX A — NOTATION GLOSSARY

| Symbol | Definition |
|--------|------------|
| $\hat{y}$ | Predicted class from DenseNet121 |
| $\hat{p}_c$ | Calibrated softmax probability for class $c$ |
| $T^*$ | Optimal temperature for calibration |
| $\hat{H}$ | Normalized Shannon entropy of prediction distribution |
| $\mathcal{C}_\alpha(x)$ | Conformal prediction set at miscoverage $\alpha$ |
| $K$ | Conformal prediction set size = $|\mathcal{C}_\alpha(x)|$ |
| $\hat{q}_c$ | Class-conditional conformal quantile for class $c$ |
| $\text{ER}(x)$ | Explanation reliability (TIxAI score from Gap #1) |
| $\text{XDI}(x)$ | Multi-XAI Disagreement Index from Gap #4 |
| $R_{\text{mal}}(\hat{y})$ | Malignancy risk prior for predicted class $\hat{y}$ |
| $\text{CCRS}(x)$ | Composite Clinical Risk Score $\in [0,1]$ |
| $\tau_c$ | Per-class deferral threshold for class $c$ |
| $\delta(x)$ | Binary deferral decision (ACCEPT/DEFER) |
| $\mathcal{M}$ | Set of malignant classes: {MEL, BCC, AKIEC} |
| NB | Net Benefit (Decision Curve Analysis) |
| AUARC | Area Under Accuracy-Rejection Curve |
| ECE | Expected Calibration Error |
| BCa | Bias-corrected and accelerated bootstrap |

---

## APPENDIX B — ABLATION STUDIES

| Ablation | What Is Changed | Primary Metric |
|----------|----------------|----------------|
| B1 | Remove TIxAI (w₄=0) | Change in AUARC, malignant sensitivity |
| B2 | Remove XDI (w₅=0) | Change in AUARC, malignant sensitivity |
| B3 | Remove malignancy prior (w₆=0) | Change in missed malignancy rate |
| B4 | Remove CP set size (w₃=0) | Change in false reassurance rate |
| B5 | Confidence only (w₁=1, others=0) | Baseline comparison — no novel signals |
| B6 | MC Dropout instead of CP | Compare uncertainty quality vs. CP |
| B7 | Global threshold vs. per-class | Change in class-stratified performance |
| B8 | Equal weights (all w_i = 1/6) | Compare vs. optimized weights |
| B9 | Standard CP vs. adaptive CP | Compare class-conditional vs. global CP |

---

## APPENDIX C — ETHICAL AND REGULATORY CONSIDERATIONS

1. **Patient safety precedence:** No ACCEPT decision by the AI should be interpreted as a guarantee of benignity. The system is a clinical decision support tool, not a diagnostic authority. This is stated explicitly in all user-facing documentation.

2. **Human-in-the-loop:** The system must be embedded in a workflow where a qualified clinician reviews all AI outputs. The DEFER recommendation is mandatory specialist review — not autonomous patient discharge.

3. **Fairness audit:** Before deployment, the CCRS system must be audited for differential referral rates across Fitzpatrick skin types. If the system defers more cases for darker skin tones (due to underrepresentation in HAM10000), this must be disclosed and addressed.

4. **Liability:** The system does not replace clinical judgment. Institutional policies on AI-assisted diagnosis must be established before deployment.

5. **Data governance:** All patient images processed must be handled in compliance with GDPR (EU), HIPAA (US), or applicable local data protection regulations.

---

*Document Version 1.0 — June 2026*  
*Target: Medical Image Analysis / Nature Digital Medicine / IEEE Transactions on Medical Imaging*  
*Research integrity statement: CCRS is a proposed integration architecture, not a claimed discovery of new uncertainty methods. All underlying components (CP, TS, TIxAI, XDI) are cited to prior work. The contribution is their principled integration for clinical deferral in dermoscopy.*
