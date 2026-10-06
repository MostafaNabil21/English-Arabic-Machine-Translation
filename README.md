# 🌍 English ↔ Arabic Neural Machine Translation

### 🧠 Skills & Technologies

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/MarianMT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NLP-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Neural%20Machine%20Translation-6A5ACD?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Fine--Tuning-7E57C2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Seq2Seq-3949AB?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face%20Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/BLEU-E91E63?style=for-the-badge" />
  <img src="https://img.shields.io/badge/chrF-C2185B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TER-AD1457?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Error%20Analysis-6D4C41?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

---

> **An end-to-end Neural Machine Translation project for English ↔ Arabic translation using fine-tuned MarianMT Transformer models, covering data analysis, preprocessing, model training, evaluation, error analysis, and interactive deployment.**

---

## 📌 Overview

This project develops a **bidirectional English ↔ Arabic Neural Machine Translation (NMT)** system using pretrained **MarianMT Transformer models** from Hugging Face.

Two separate MarianMT models are fine-tuned for the two translation directions:

* 🇬🇧 **English → Arabic**
* 🇪🇬 **Arabic → English**

The project follows an end-to-end NLP workflow, from preparing and analyzing the bilingual corpus to fine-tuning Transformer models, evaluating translation quality, inspecting translation errors, and deploying the models through an interactive Gradio interface.

---

## 🎯 Objectives

The main objectives of this project are to:

* Build an English ↔ Arabic Neural Machine Translation system.
* Fine-tune pretrained MarianMT Transformer models.
* Work with a parallel English-Arabic corpus.
* Perform exploratory data analysis.
* Clean and deduplicate bilingual sentence pairs.
* Tokenize the dataset for Transformer-based training.
* Evaluate translation quality using multiple metrics.
* Analyze individual translation outputs.
* Build a reusable inference layer.
* Deploy the trained models through a Gradio interface.

---

## 🧠 Models

The project uses two independently fine-tuned MarianMT models.

| Translation Direction      | Pretrained Model             |
| -------------------------- | ---------------------------- |
| 🇬🇧 English → 🇪🇬 Arabic | `Helsinki-NLP/opus-mt-en-ar` |
| 🇪🇬 Arabic → 🇬🇧 English | `Helsinki-NLP/opus-mt-ar-en` |

Each model is trained specifically for its corresponding translation direction.

---

## 📊 Dataset

The project uses the Hugging Face parallel corpus:

**`ymoslem/UN-Arabic-English-Filtered`**

The notebook uses a configurable number of training pairs:

```python
N_TRAIN_PAIRS = 50_000
```

The data is divided into:

* Training set
* Development / validation set
* Test set

The English and Arabic sentences are aligned as parallel translation pairs.

---

## 🔍 Exploratory Data Analysis

Before training the models, the notebook performs exploratory analysis of the bilingual corpus.

The analysis includes:

### Data Quality

* Dataset split sizes
* Empty sentence detection
* Duplicate translation-pair detection
* Sentence statistics

### Sentence Analysis

* English sentence lengths
* Arabic sentence lengths
* Word-count distributions
* English vs. Arabic sentence-length relationships
* Arabic-to-English word-count ratios

### Vocabulary Analysis

The notebook also examines frequently occurring words in both languages to better understand the corpus.

Visualizations are created using **Matplotlib**.

---

## 🧹 Data Cleaning & Preprocessing

The project applies preprocessing before model training.

The cleaning pipeline:

1. Removes empty English sentences.
2. Removes empty Arabic sentences.
3. Removes duplicate English-Arabic translation pairs.
4. Resets the dataset index.
5. Produces clean datasets for tokenization.

Example:

```python
def clean_pairs(dataset):
    dataset = dataset.filter(
        lambda x: bool(x["text_en"].strip())
        and bool(x["text_ar"].strip())
    )

    pdf = (
        dataset.to_pandas()
        .drop_duplicates(
            subset=["text_en", "text_ar"]
        )
        .reset_index(drop=True)
    )

    return Dataset.from_pandas(
        pdf,
        preserve_index=False
    )
```

---

## 🔤 Tokenization

Each MarianMT model uses its corresponding pretrained tokenizer.

The notebook limits the sequence length to:

| Parameter             | Value |
| --------------------- | ----: |
| Maximum source tokens |   128 |
| Maximum target tokens |   128 |

The bilingual text is converted into tokenized inputs and target labels suitable for sequence-to-sequence Transformer training.

---

## ⚙️ Fine-Tuning

The MarianMT models are fine-tuned using **Hugging Face Transformers** and **PyTorch**.

### Training Configuration

| Parameter             |  Value |
| --------------------- | -----: |
| Training pairs        | 50,000 |
| Epochs                |     10 |
| Training batch size   |      8 |
| Evaluation batch size |      8 |
| Gradient accumulation |      4 |
| Learning rate         | `3e-5` |
| Maximum source tokens |    128 |
| Maximum target tokens |    128 |
| Random seed           |     42 |

The notebook also detects whether CUDA is available:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()
```

This allows the training workflow to take advantage of GPU acceleration when available.

---

## 📈 Evaluation

The translation models are evaluated using several automatic machine-translation metrics.

### BLEU

**BLEU** measures n-gram overlap between the generated translation and the reference translation.

**Higher scores indicate greater similarity to the reference.**

### chrF

**chrF** evaluates character-level similarity between the generated translation and the reference.

**Higher scores indicate better similarity.**

### TER

**TER (Translation Edit Rate)** measures the number of edits required to transform the generated translation into the reference.

**Lower scores indicate better performance.**

### Metrics Used

```text
BLEU
chrF
TER
```

Using multiple metrics provides a broader view of translation quality.

---

## 📉 Training Analysis

The notebook tracks model training and evaluation behavior throughout fine-tuning.

Training diagnostics include:

* Training loss
* Validation loss
* Model performance across training steps

These visualizations help monitor the learning process and identify potential convergence or overfitting issues.

---

## 🔎 Qualitative Evaluation & Error Analysis

In addition to automatic metrics, the project performs a lightweight qualitative analysis of model predictions.

For each MarianMT model:

1. A sample of test examples is selected.
2. The fine-tuned model generates translations.
3. Sentence-level BLEU scores are calculated.
4. Predictions are ranked by score.
5. The lowest-scoring examples are displayed for inspection.

Example analysis structure:

```text
Source
   ↓
Reference Translation
   ↓
Model Prediction
   ↓
Sentence-Level BLEU
   ↓
Lowest-Scoring Examples
```

This makes it possible to inspect translation behavior beyond the overall evaluation metrics.

---

## 🔄 Reusable Translation Service

The notebook implements a reusable `TranslationService` class for model inference.

Example:

```python
translator = TranslationService(
    model_path,
    model_type="marian",
    src_lang="en",
    tgt_lang="ar"
)

translator.translate(
    "Hello, how are you?"
)
```

The service:

* Loads the trained MarianMT model.
* Loads the corresponding tokenizer.
* Automatically uses the configured device.
* Tokenizes input text.
* Generates translations.
* Returns decoded translation text.

---

## 🖥️ Interactive Gradio Application

The project includes a **Gradio interface** for interactive translation.

### Supported Directions

```text
🇬🇧 English → 🇪🇬 Arabic

🇪🇬 Arabic → 🇬🇧 English
```

The interface provides:

* Source text input
* Translation direction selection
* Generated translation output
* Browser-based interaction

The Gradio interface is powered by the two fine-tuned MarianMT models.

---

## 🔄 End-to-End Workflow

```text
                ┌────────────────────────┐
                │   Parallel Corpus      │
                │   English ↔ Arabic     │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Exploratory Data       │
                │ Analysis               │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Data Cleaning &         │
                │ Deduplication           │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Tokenization            │
                └───────────┬────────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
       ┌──────────────────┐   ┌──────────────────┐
       │ MarianMT         │   │ MarianMT         │
       │ English → Arabic │   │ Arabic → English │
       └────────┬─────────┘   └────────┬─────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                ┌────────────────────────┐
                │ Model Evaluation       │
                │ BLEU / chrF / TER      │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Qualitative Evaluation │
                │ & Error Analysis       │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Translation Service    │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Gradio Web Interface   │
                └────────────────────────┘
```

---

## 🛠️ Technology Stack

### Programming & Frameworks

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Jupyter Notebook

### NLP & Deep Learning

* Natural Language Processing
* Neural Machine Translation
* Transformer Models
* Sequence-to-Sequence Learning
* Fine-Tuning
* Tokenization
  
### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Data Cleaning
* Exploratory Data Analysis

### Evaluation

* BLEU
* chrF
* TER
* Sentence-Level Evaluation
* Qualitative Error Analysis

### Deployment

* Gradio
* Model Inference
* Interactive Machine Learning Application

---

## 📦 Installation


Install the required packages:

```bash
pip install torch transformers datasets sacrebleu pandas numpy matplotlib gradio
```

---

## ⚡ GPU Support

The notebook automatically checks for CUDA availability.

If a compatible GPU is available:

```python
DEVICE = "cuda"
USE_FP16 = True
```

Otherwise, the notebook falls back to CPU execution.

GPU acceleration is recommended for faster Transformer fine-tuning.

---

## 🚀 Future Improvements

Potential future improvements include:

* Training with larger portions of the parallel corpus.
* Hyperparameter optimization.
* Experimenting with different training configurations.
* Improving Arabic-specific preprocessing.
* Analyzing translation quality by sentence length.
* Adding additional evaluation metrics.
* Performing human evaluation.
* Model quantization for more efficient inference.
* Deploying the translation service as an API.
* Deploying the Gradio application permanently.

---

## 📚 What This Project Demonstrates

This project demonstrates practical experience with an end-to-end **Natural Language Processing and Deep Learning workflow**:

```text
Dataset Preparation
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Tokenization
        ↓
Transformer Fine-Tuning
        ↓
Model Evaluation
        ↓
Qualitative Analysis
        ↓
Inference
        ↓
Interactive Deployment
```

The project combines **NLP, Transformer models, PyTorch, Hugging Face, data preprocessing, model fine-tuning, evaluation, and deployment** into a complete machine-translation pipeline.
