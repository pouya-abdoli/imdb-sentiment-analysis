# 🎬 IMDB Sentiment Analysis — Classic ML vs. Deep Learning

A hands-on comparison of two very different approaches to binary sentiment classification on the [IMDB movie review dataset](https://huggingface.co/datasets/imdb): a lightweight **TF-IDF + Logistic Regression** baseline versus a fine-tuned **DistilBERT** transformer.

The goal isn't to prove one approach "wins" — it's to measure the actual trade-off between **model size**, **training time**, and **accuracy**, and to make the case for always building a simple baseline before reaching for deep learning. ⚖️

---

## 📊 Results at a Glance

| | 🧮 TF-IDF + Logistic Regression | 🤖 DistilBERT (fine-tuned) |
|---|---|---|
| **Parameters** | ~10,000 | ~66,000,000 |
| **Pipeline time** (load + vectorize + train) | **< 30 sec** ⚡ | **~45 min** ⏱️ |
| **Accuracy** | 88.2% | 92.9% |
| **F1 score** | 0.883 | 0.929 |
| **Recall** | 0.885 | 0.933 |

**Takeaway:** DistilBERT is ~6,600× larger and takes roughly **100× longer to train**, in exchange for about **4.7 points of accuracy**. For a lot of real-world use cases, that trade isn't worth it. 🎯

> 💡 **Lesson learned:** Deep learning models are famously "data hungry" and keep improving as you feed them more data — which is exactly why it's tempting to assume they're always the better choice. But before reaching for a heavyweight transformer, it pays to build a simple, fast baseline first. Often the extra accuracy from a bigger model doesn't justify the extra time, compute cost, and complexity — especially under tight resource or latency constraints.
>
> This isn't an argument that deep learning is "bad" or that classic ML is "better" — it's a reminder to always ask: *does my problem actually need this level of complexity?* ✅

---

## 🗂️ Project Structure

```
imdb-sentiment-analysis/
├── TF‑IDF + Logistic Regression.ipynb   # Classic ML baseline
├── distilbert-imdb.ipynb                # Fine-tuned transformer model
└── README.md
```

---

## 🧮 Model 1: TF-IDF + Logistic Regression

A fast, fully classical NLP pipeline:

- **Vectorization:** `TfidfVectorizer` — 10,000 max features, unigrams + bigrams, English stopwords removed
- **Classifier:** `LogisticRegression` (scikit-learn), max 1,000 iterations
- **Data:** IMDB `train` / `test` split (25,000 / 25,000 reviews) loaded via 🤗 `datasets`

**Measured runtime:**
| Step | Time |
|---|---|
| Dataset load | 1.20 s |
| TF-IDF vectorization | 25.39 s |
| Model training | 1.15 s |
| **Total** | **~27.7 s** |

**Test set performance:**
- Accuracy: **0.882**
- Precision: **0.880**
- Recall: **0.885**
- F1 score: **0.883**

---

## 🤖 Model 2: DistilBERT (fine-tuned)

A transformer-based approach using 🤗 `transformers`:

- **Base model:** `distilbert-base-uncased`
- **Training setup:** 2 epochs, learning rate `2e-5`, batch size 32 (train) / 16 (eval)
- **Data split:** 22,500 train / 2,500 validation (90/10 split of the training set), evaluated on the full 25,000-review test set

**Measured runtime:**
- Training: **2,716.6 s (≈ 45.3 minutes)** for 1,408 steps
- Full test-set inference: **387.3 s (≈ 6.5 minutes)**

**Test set performance:**
- Accuracy: **0.929**
- F1 score: **0.929**
- Recall: **0.933**

Both notebooks include a confusion matrix visualization for a closer look at error types (false positives vs. false negatives).

---

## 🚀 Getting Started

```bash
git clone https://github.com/pouya-abdoli/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install datasets scikit-learn transformers evaluate matplotlib torch
jupyter notebook
```

Then open either notebook and run all cells. The DistilBERT notebook benefits significantly from a GPU — expect much longer training times on CPU.

---

## 🧠 Why This Comparison Matters

It's easy to assume that bigger, more modern models are always the right call. This project is a small, concrete counter-example: a **6,600× smaller model** trained in **under 30 seconds** gets within **~5 percentage points** of a transformer that took **45 minutes** to fine-tune.

Before committing to a deep learning solution, it's worth asking:

- Does the accuracy gain actually matter for my use case?
- Can I afford the training time and compute cost?
- Do I need low-latency inference?
- Would a simple baseline already be "good enough"?

Building the cheap baseline first isn't a step to skip — it's often the fastest way to find out whether the expensive model is even necessary. 🧭

---

## 📄 License

No license specified yet — feel free to open an issue if you'd like to use this work and need clarification.
