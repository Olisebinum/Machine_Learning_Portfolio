# Models Folder

This folder contains all **trained machine learning models** for the Supervised Learning Personal Project.  

The models are saved in formats such as **Pickle (`.pkl`)** or **Joblib (`.joblib`)**, making them reusable, shareable, and ready for deployment.

---

## Contents

1. **Linear Models**
   - Simple Linear Regression
   - Multiple Linear Regression
   - Ridge and Lasso Regression
   - Used for baseline and regularized predictions

2. **Tree-Based Models**
   - Decision Tree Regressor
   - Random Forest Regressor
   - Models trained using ensemble techniques to reduce variance and improve accuracy

3. **Ensemble Models**
   - Bagging Regressor
   - Random Forest
   - Combining multiple base models for better generalization

---

## Purpose

- **Reproducibility:** Allows anyone to reload trained models without retraining from scratch.  
- **Efficiency:** Saves time and computational resources by reusing already trained models.  
- **Deployment-Ready:** Models are saved in formats compatible with production pipelines.  
- **Experiment Tracking:** Organizes multiple models for comparison and iterative improvement.

---

## Notes for Users

- Model naming conventions:
  - `linear_regression.pkl` → Simple Linear Regression
  - `ridge_model.pkl` → Ridge Regression
  - `lasso_model.pkl` → Lasso Regression
  - `decision_tree.pkl` → Decision Tree Regressor
  - `random_forest.pkl` → Random Forest Regressor
- Always ensure that the **same preprocessing steps** are applied to new data before using these models.  
- Models can be loaded using Python:

```python
import joblib

model = joblib.load('models/random_forest.pkl')
predictions = model.predict(new_data)

