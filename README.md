# model-bias-audit-demo

**Detecting and mitigating demographic bias in a credit scoring model** — a practical fairness audit using industry-standard metrics and two mitigation techniques.

Credit scoring models are classified as **high-risk AI** under the EU AI Act and subject to explicit bias assessment requirements under OSFI E-23 and SR 11-7. This notebook demonstrates what a bias audit looks like in practice: what to measure, how to visualise it, and what to do about it.

---

## What It Does

1. **Generates** a synthetic consumer lending dataset with structural demographic bias reflecting historical discrimination encoded in training data
2. **Trains** a baseline logistic regression model and audits it across four fairness metrics
3. **Visualises** the bias — approval rate gaps, ROC curves by group, score distributions
4. **Applies two mitigation techniques** and compares results

---

## Fairness Metrics

| Metric | Definition | Regulatory relevance |
|---|---|---|
| Demographic Parity | Equal approval rates across groups | CFPB disparate impact doctrine |
| Equal Opportunity | Equal true positive rates for qualified applicants | Most fairness-focused regulation |
| Predictive Parity | Equal precision across groups | OSFI E-23, SR 11-7 validation |
| Disparate Impact Ratio | Approval rate ratio ≥ 0.8 (4/5ths rule) | US EEOC, CFPB lending rules |

> A model can achieve equal overall accuracy while failing every fairness metric. Accuracy is not a fairness measure.

---

## Mitigation Techniques

| Technique | Stage | How it works |
|---|---|---|
| **Reweighing** | Pre-processing | Upweights underrepresented (group, outcome) combinations in training |
| **Threshold Optimisation** | Post-processing | Applies per-group decision thresholds to equalise true positive rates |

---

## Key Findings

- Baseline model fails the 4/5ths disparate impact threshold despite reasonable overall accuracy
- Reweighing closes ~50% of the approval rate gap with negligible accuracy cost
- Threshold optimisation achieves near-equal opportunity but requires group membership at inference time
- No single technique satisfies all fairness metrics simultaneously — tradeoffs must be documented and justified

---

## Regulatory Context

- **EU AI Act (Annex III):** Credit scoring is explicitly listed as high-risk. Article 9 (risk management) and Article 10 (data governance) require bias testing and mitigation
- **OSFI E-23:** Model validation must include assessment of disparate impact for models affecting protected groups
- **SR 11-7 (US Federal Reserve):** Outcomes analysis requirement covers unintended consequences including disparate impact

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nbformat
jupyter notebook model_bias_audit.ipynb
```

No external data downloads required — dataset is synthetically generated.

---

## Extensions

- Apply to real-world data: [HMDA mortgage data](https://www.consumerfinance.gov/data-research/hmda/), [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)
- Add intersectional analysis across multiple protected attributes
- Implement adversarial debiasing (in-processing mitigation)
- Build a fairness monitoring dashboard for production deployment

---

*Author: [Meng-Yan Chen](https://github.com/meng-yanchen)*  
*Related: [nlp-regulatory-text-analysis](https://github.com/meng-yanchen/nlp-regulatory-text-analysis) · [ai-governance-tracker](https://github.com/meng-yanchen/ai-governance-tracker)*
