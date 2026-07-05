# GAP #6 — TEMPORAL/STOCHASTIC SALIENCY STABILITY (STS)
# Is the Explanation Reproducible, Independent of Whether It's Correctly Localized?

### **When Should Clinicians Not Trust an AI Explanation?**
> A Trust-Aware Framework for Dermoscopic Skin Cancer Diagnosis
> **Contribution 4 (revised): Explanation Reproducibility Under Stochastic Re-Inference**

---

> **Document Type:** Results Report (post-execution)
> **Date:** 2026-07-05
> **Status:** This document fully replaces the previous version of `GAP6_Longitudinal_Consistency_Report.md`, which described a **different, never-implemented study** — multi-visit longitudinal saliency drift across repeated clinic visits of the same patient, using external datasets ("UQ Longitudinal Dataset," "Basel SDDI1") that do not exist anywhere in this project. That document's own Section 2 feasibility audit had already concluded HAM10000 itself is "Unsuitable" for that specific study, since HAM10000 is not a structured longitudinal follow-up dataset. This file now instead documents **GAP-6 as the notebook actually defines it** (CELL 10c's markdown title, updated 2026-07-05 to read "GAP-6: Temporal/Stochastic Saliency Stability (STS)"): a test-retest reproducibility check using repeated *stochastic* (MC-Dropout) inference on the *same* image, not repeated clinical visits over calendar time. If a genuine multi-visit longitudinal study is wanted in the future, it needs a different dataset than anything currently in this project and should get a new, distinctly-numbered report rather than reusing "GAP-6."

---

## 1. The Scientific Question

TIxAI (GAP-1) asks: *"is the explanation localized in the right place?"* It says nothing about whether that same explanation would look the same if you asked the model again, right now, with nothing about the input changed.

> **If you run the exact same image through the exact same trained model twice, holding the predicted class fixed, does Grad-CAM++ produce the same saliency map both times — or does the explanation itself move around due to the model's own internal stochasticity (dropout)?**

This matters independently of localization quality: an explanation can be *well-localized on average* but *unstable* — pointing at slightly different sub-regions of the lesion (or, worse, drifting off it entirely) from one inference call to the next. A clinician shown a single Grad-CAM++ overlay has no way to know, from that one image, whether it's a stable read of the model's reasoning or one noisy draw from a wide distribution of possible explanations for the same input.

## 2. Method (as actually implemented, CELL 10c)

For each of 100 randomly sampled test images:
1. Run one deterministic forward pass (dropout off) to get the predicted class — this class is held fixed for all subsequent CAMs, so any variation measured is due to stochasticity in the *explanation*, not a changing prediction.
2. Enable MC-Dropout (dropout layers active/`.train()`) and generate `T=15` Grad-CAM++ maps for that same image and that same predicted class, each with fresh dropout noise.
3. Flatten each of the 15 maps and compute the full pairwise Pearson correlation matrix between them; average the upper-triangle entries (the 105 unique pairs) into a single **STS score** for that image, in [-1, 1] (in practice, [0, 1]-ish since CAMs are non-negative).
4. STS close to 1.0 = the 15 stochastic explanations are nearly identical (reproducible). STS close to 0 or negative = the explanation itself is unstable — different dropout noise draws produce meaningfully different saliency maps for the same image and prediction.

No lesion segmentation mask is required for this metric, unlike TIxAI — it is a pure self-consistency check.

## 3. Real Result (n=100 test images, T=15 passes each)

**Median STS: 0.850. Mean STS: 0.797.**

**7 of 100 images (7.0%) have STS < 0.5** — for these, the explanation's location is not reproducible across repeated stochastic passes, *regardless of what TIxAI or accuracy say about that same image*. A clinician could be shown a confident, well-localized-looking heatmap for one of these cases and have no way to know a second inference call might highlight a visibly different region.

**Relationship to TIxAI** (Spearman correlation, n=44 images with both scores available): **ρ = −0.410, p = 0.0057** — a moderate, statistically significant negative correlation. Higher explanation stability is associated with *lower* TIxAI in this sample, which is a genuinely counter-intuitive result worth flagging rather than glossing over: naively one might expect a well-localized (high-TIxAI) explanation to also be a stable one. Two non-exclusive readings:
- The correlation is moderate (|ρ|≈0.41), not high — STS and TIxAI are measuring related but distinct properties ("where" vs. "how consistently"), consistent with the project's original framing that STS should not be redundant with TIxAI.
- n=44 is a small overlap sample (STS was computed on a separate random 100-image draw from TIxAI's 466 correctly-classified images, intersected here); a negative sign this size on n=44 should be treated as a signal worth re-checking at larger n, not yet a settled mechanistic finding, before building strong clinical claims on the direction of the relationship.

## 4. Caveats

- Like every other DenseNet121-dependent result in this project, this reflects a checkpoint trained to only 60/150 epochs under a since-fixed LR-scheduler bug (`GAP1_Model_Development_Report.md` §2.5) — re-run after the full retrain to confirm the 7% instability rate and the TIxAI correlation direction hold.
- STS uses MC-Dropout as its stochasticity source. `CELL 17b`'s cross-architecture audit notes that `timm`'s `convnext_tiny` has no active `nn.Dropout` modules by default (it uses stochastic depth instead), so this exact STS methodology does not transfer to ConvNeXt-Tiny without modification — this is a DenseNet121-specific result, not yet a cross-architecture one.
- "Temporal" in the metric's name refers to repeated *inference-time* stochastic passes on a static image, not the passage of calendar time or repeated clinical visits — this document's title and framing have been corrected accordingly (see status note above).

## 5. What Would Be Needed for a Genuine Multi-Visit Longitudinal Study

The retired version of this document raised a real and reasonable research question — does a model's explanation stay consistent when the *same patient's same lesion* is re-imaged at a later clinical visit, months apart, with the lesion itself possibly having changed? That question is still open and still worth pursuing, but it requires:
- A dataset with genuine repeated-visit imaging of the same lesion over time (HAM10000 does not have this structure), and
- Spatial registration between visit images (the retired document's SIFT/RANSAC approach was a reasonable proposal), and
- A new, distinct GAP number and report — it should not be conflated with the stochastic-reproducibility result documented here, since the two questions ("does the explanation move under dropout noise right now" vs. "does the explanation move as the lesion changes over months") require different data, different confounds, and different statistical designs.

## Appendix: Notation

| Symbol | Definition |
|---|---|
| STS | Saliency Temporal Stability score: mean pairwise Pearson correlation across T stochastic Grad-CAM++ passes on one image |
| $T$ | Number of stochastic passes per image (15 in this run) |
| UNSTABLE_THRESHOLD | 0.5 — below this, an image's explanation is flagged non-reproducible |
