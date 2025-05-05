# DeLUCS – Final Project (Modified)

This repository is a **modified and extended version of [DeLUCS](https://doi.org/10.1371/journal.pone.0261531)** – an unsupervised deep learning framework for clustering DNA sequences.

Our final project involved applying DeLUCS to new datasets, extending its architecture, and evaluating its performance.

---

## 🔧 Summary of Modifications

We made the following changes to the original DeLUCS codebase:

- 📁 Added `compare.py`: a new script to **compare the performance of DeLUCS with an improved variant** using multiple attention heads (`Net_linear_improved`).
- 🧠 Implemented `Net_linear_improved` with a configurable number of heads for richer representations.
- 📊 Logged training loss and test accuracy over multiple runs and visualized comparisons.
- 🧪 Evaluated on the **Bacteria** dataset (`data/Bacteria`) with 3 variants: Original, Improved-8-heads, Improved-16-heads.
- 📄 Added our final report summarizing methodology, results, and conclusions: [`src/final_report.pdf`](src/final_report.pdf)

---

## 🗂 Structure

- `src/EvaluateDeLUCS.py`: Original training and evaluation code.
- `src/compare.py`: Our modified evaluation pipeline comparing different architectures.
- `src/PytorchUtils.py`: Includes `Net_linear` and our `Net_linear_improved`.
- `data/Bacteria`: Dataset used for the extended analysis.

---

## 📊 Results

See the full summary and visualizations in the [📄 final report (PDF)](src/final_report.pdf).

Key outcomes:
- Improved model with 16 heads achieved better clustering accuracy across multiple runs.
- Training stability was also enhanced, as seen in the loss curves.

---

## 💡 How to Run

To compare the original vs. improved models on the Bacteria dataset:

```bash
python src/compare.py
