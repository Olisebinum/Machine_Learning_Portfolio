# Data Folder

This folder contains all datasets used for the Supervised Learning Personal Project.  

It is structured to support **end-to-end machine learning workflows**, from raw data exploration to final model training.  

## Contents

1. **Raw Data**
   - Original datasets collected from sources (e.g., CSV, Excel, JSON)
   - Unprocessed data used for initial exploration
   - Always kept unchanged to ensure reproducibility

2. **Processed Data**
   - Cleaned and transformed versions of raw datasets
   - Missing values handled, categorical variables encoded
   - Features scaled or normalized if necessary
   - Split into training, validation, and test sets

3. **Feature Data**
   - Datasets after feature engineering
   - Includes derived columns, interaction features, or aggregated statistics
   - Ready for feeding into machine learning models

## Purpose

- **Reproducibility:** Ensures anyone can retrace the data processing steps and rebuild the models.  
- **Clarity:** Separates raw data from processed and feature-ready datasets for easier workflow management.  
- **Scalability:** Prepares the project for future expansion (e.g., adding new datasets, testing new features).

## Notes for Users

- Do not overwrite the raw datasets. Always work on copies in the processed folder.  
- Ensure consistent formatting when adding new datasets.  
- Naming conventions:
  - `raw_<dataset_name>.csv` → raw dataset
  - `processed_<dataset_name>.csv` → cleaned dataset ready for modeling
  - `features_<dataset_name>.csv` → dataset with engineered features

---

> This folder is a **central component** of the supervised learning pipeline. Proper organization here ensures smooth model building, evaluation, and reproducibility.

