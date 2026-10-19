<div align="center">

# 🔁 Recurrent Small Language Models

### *Architecture, Training, and Evaluation — A Technical Report*

[![Report PDF](https://img.shields.io/badge/PDF-Technical_Report-B08D3E?style=for-the-badge&logo=adobeacrobatreader)](report/Recurrent_Small_LMs_Technical_Report_NeurIPS.pdf)
[![Pages](https://img.shields.io/badge/22_pages-two_column-23201A?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/review-9/10-brightgreen?style=for-the-badge)]()

**One honest question, answered with real numbers:** *do recurrent transformers gain anything from running extra loops at test time?*

**Answer: no.** Perplexity cliff-dives from 29.4 → 4.4 as loops go 1 → 4 (the trained depth), then flatlines — 8, 12, 16 loops buy nothing. With 95% bootstrap confidence intervals to prove it.

</div>

---

## 📊 The centrepiece result

![Loop-depth sweep with 95% bootstrap CIs](figures/fig1_sweep_curve.png)

| Loops | 1 | 2 | **4 (trained)** | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| Perplexity | 29.38 | 11.18 | **4.41** | 4.90 | 4.91 | 4.91 |
| 95% CI | [28.30, 30.52] | [10.84, 11.54] | **[4.28, 4.55]** | [4.75, 5.06] | [4.76, 5.07] | [4.76, 5.07] |

The 4→8 degradation is real: paired per-sequence difference **+0.106 loss, CI [+0.101, +0.110]**. Test-time loop scaling saturates at the trained depth and slightly degrades beyond it.

### 🔥 In 3D

![3D saturation surface: both metrics cliff-dive to loop 4, then flatline](figures/fig4_3d_saturation.png)

---

## 🏗️ The reference model

![OpenMythos-style 44M architecture: prelude, loop block with MoE experts, coda](figures/fig2_architecture.png)

```mermaid
flowchart LR
    A[Input tokens] --> B[Prelude<br/>6 layers]
    B --> C{Loop block ×4<br/>8 experts · MoE<br/>GQA attention}
    C --> D[Coda<br/>2 layers]
    D --> E[Logits]
    style C fill:#EBDEC0,stroke:#B08D3E,stroke-width:3px
```

- **43,974,722 parameters** · dim 512 · GPT-2 BPE (50,257)
- Trained on TinyStories, 120M tokens, loss 9.52 → 1.53

---

## 🗺️ Why GQA matters: the KV-cache heatmap

![KV-cache heatmap across batch size and context length](figures/fig3_kv_heatmap.png)

At batch 8 / 2k context the MHA cache (403 MB) is **4× the weights** — GQA reclaims ~300 MB. At heavy batching it's the difference between fitting on one GPU or not.

---

## 📖 What's inside the report

| Section | Topic | Status |
|---|---|---|
| 3 | **Loop-depth extrapolation sweep** (the executed study) | ✅ Measured + bootstrap CIs |
| 4 | MoE routing dynamics · GQA efficiency · depth–width tradeoffs | 📐 Analytical |
| 5 | Curriculum protocol · loop-aware distillation | 📝 Proposed |
| 6 | Test-time compute taxonomy · quantization · mechanistic probes · TinyLoop-Bench | 📐/📝 Mixed |
| 7 | Conclusion + the key next experiment | 🎯 |

**Honesty rule:** proposed experiments are labelled as proposed. The only executed study is the loop sweep. CIs capture sampling noise over sequences — run-to-run seed variance is named as the next experiment.

---

## 📁 Repo layout

```
├── report/     → the 22-page two-column PDF + full LaTeX sources
├── figures/    → all diagrams (PNG)
└── scripts/    → make_figures.py (regenerate every figure from the data)
```

Regenerate figures: `python3 scripts/make_figures.py` (needs matplotlib + numpy).

---

<div align="center">

*Built from a real training run — 4,850 steps on a Kaggle T4, October 2026.*
**Jayant Dubey** · October 2026

</div>
