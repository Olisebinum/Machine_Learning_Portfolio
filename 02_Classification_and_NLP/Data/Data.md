# 📁 Data

## 📌 Overview

This folder contains all datasets used in the **02_Classification_and_NLP** module. It includes both **raw** and **processed** data used for building, training, and evaluating classification and NLP models.

The purpose of this folder is to clearly separate data handling from modeling logic, ensuring **reproducibility, transparency, and ease of review** for recruiters and collaborators.

---

## 📂 Folder Structure

```
data/
│
├── raw/                   # Original datasets as obtained from the source
│                           # These files are not modified
│
├── processed/             # Cleaned and transformed datasets ready for modeling
│                           # Includes encoded, scaled, or vectorized data
│
├── text/                  # Text-based datasets used for NLP tasks
│                           # Examples: reviews, tweets, documents
│
├── external/              # Optional third-party or reference datasets
│
└── README.md              # Documentation describing data sources and usage
```

---

## 🗂️ Data Categories Explained

### 🔹 Raw Data (`raw/`)

Contains the original datasets exactly as they were collected or downloaded.

**Purpose:**

* Preserve data integrity
* Enable traceability and reproducibility
* Allow reprocessing from scratch if needed

**Typical Contents:**

* CSV files
* Text files
* Exported survey or system data

> ⚠️ Raw data files are **never edited directly**.

---

### 🔹 Processed Data (`processed/`)

Contains datasets that have been cleaned and transformed for machine learning.

**Typical Processing Steps:**

* Handling missing values
* Encoding categorical variables
* Feature scaling and normalization
* Train-test splits
* Vectorized text features (BoW / TF-IDF)

**Purpose:**

* Speed up experimentation
* Ensure consistent inputs across models

---

### 🔹 Text Data (`text/`)

Contains unstructured text datasets used for NLP experiments.

**Examples:**

* Customer reviews
* Social media posts
* Short documents or comments

**Usage:**

* Text preprocessing and normalization
* Feature extraction (BoW, TF-IDF)
* Sentiment analysis and text classification

---

### 🔹 External Data (`external/`)

Optional folder for datasets sourced from third parties or public repositories.

**Examples:**

* Kaggle datasets
* Open government data
* Academic or benchmark datasets

Each external dataset should include a reference to its source.

---

## 🧠 How the Data Is Used

Across the notebooks in this module, datasets are used in the following workflow:

1. Load raw or text data
2. Perform exploratory data analysis (EDA)
3. Apply cleaning and preprocessing steps
4. Save processed versions for reuse
5. Train and evaluate classification or NLP models

This approach mirrors **industry-standard ML pipelines**.

---

## 📌 Notes for Reviewers

* Data preprocessing steps are fully documented inside the notebooks
* Processed datasets are reproducible from raw data
* No sensitive or private data is included

---

## ⭐ Acknowledgement

Datasets used in this project are sourced from publicly available resources or created for educational purposes.

---

**Module:** 02_Classification_and_NLP
**Author:** Olise Ebinum
**License:** MIT License

