# 📊 Lexical-Constrained Text Simplification: Analysis Scripts

This directory contains scripts to inspect, analyze, and visualize the output of the simplification benchmark evaluations.

## 📁 Folder Contents

1. **`print_statistics.py`**:
   * Runs quantitative analysis on `MASTER_BENCHMARK_MATRIX.csv`.
   * Outputs global model comparison averages (SARI, BLEU, BERTScore, Cosine Similarity).
   * Evaluates curated vs. standard frequency-based vocabularies (e.g., Randall Munroe's *Thing Explainer 1000* vs. `vocab_1000.txt`).
   * Highlights the peak-performing combinations per dataset.

2. **`generate_graphs.py`**:
   * Generates the three consolidated, non-cluttered visualizations illustrating the core findings of the paper.
   * Saves graphs to `analysis/plots/`:
     * `model_comparison.png`: Metric scaling (SARI) vs. vocabulary size for all four models.
     * `stopword_exemption.png`: Semantic preservation (BERTScore) vs. vocabulary size comparing stopword exemption configurations.
     * `tradeoffs.png`: A scatter plot evaluating the trade-offs between Grammatical Fluency (METEOR) and Meaning Preservation (BERTScore) for all architectures.


3. **`extract_examples.py`**:
   * Extracts and compares model outputs side-by-side on the Turk corpus.
   * Focuses on illustrative sentence examples (Jargon, Medical terms, History) to qualitatively show why the MLM architecture excels while word-substitution and autoregressive Seq2Seq models struggle.

---

## 🚀 How to Run

Ensure your virtual environment is active and all project dependencies are installed (e.g., `pandas`, `matplotlib`, `seaborn`, `numpy`).

Run the scripts from the terminal:

### 1. Print Metric Averages and Statistics
```bash
python analysis/print_statistics.py
```

### 2. Generate Core Evaluation Graphs
```bash
python analysis/generate_graphs.py
```
*This will create the plots in `analysis/plots/`.*

### 3. Extract Example Sentence Output Comparisons
```bash
python analysis/extract_examples.py
```
