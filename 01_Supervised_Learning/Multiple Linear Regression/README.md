# Multiple Linear Regression — Olise Ebinum

## Overview
This notebook expands my supervised machine learning skills by applying **Multiple Linear Regression (MLR)** to examine how several independent variables together influence a continuous target variable.

Unlike simple linear regression, this project explores multidimensional relationships, checks for multicollinearity, and evaluates the overall strength and reliability of the model.  
The focus is not only on prediction accuracy but also on interpreting how each feature contributes to the outcome and validating the statistical assumptions of MLR.

---

## Project Workflow

### 1. Data Exploration & Preparation
- Loaded and inspected the dataset using Pandas and NumPy.  
- Analyzed descriptive statistics, feature distributions, and correlations.  
- Visualized predictor–target relationships using pairplots, heatmaps, and scatter matrices.  
- Handled missing values and encoded categorical variables.  
- Standardized numerical features where needed.

### 2. Model Development
- Implemented **Multiple Linear Regression** using scikit-learn.  
- Split the dataset into training and testing subsets.  
- Fitted the model using multiple predictors simultaneously.  
- Extracted key parameters:
  - Model coefficients  
  - Intercept  
- Interpreted each coefficient to understand how features influence the target.

### 3. Model Evaluation & Diagnostics
- Calculated performance metrics:
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
  - R² score  
- Examined residual plots for:
  - Linearity  
  - Homoscedasticity  
  - Normality  
  - Independence  
- Checked for **multicollinearity** using Variance Inflation Factor (VIF).  
- Suggested improvements through feature selection and regularization (Ridge, Lasso).

### 4. Insights & Interpretation
- Identified features with strong or weak influence on the target.  
- Interpreted the direction and magnitude of coefficients.  
- Discussed the effect of multicollinearity on model stability.  
- Evaluated the model’s limitations and opportunities for improvement.

---

## Tools & Technologies

| Category | Tools |
|---------|--------|
| Programming & Analysis | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (Linear Regression, VIF analysis) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## Key Takeaways

- Built a complete multiple regression model from preprocessing to evaluation.  
- Gained deeper understanding of feature relationships and regression assumptions.  
- Learned how to detect and mitigate multicollinearity.  
- Improved ability to explain statistical findings clearly and professionally.  

---

## Learning Impact
This project strengthened my ability to:
- Analyze datasets with multiple predictors  
- Apply advanced regression techniques  
- Perform diagnostic checks to validate model assumptions  
- Present data-driven insights in a clear and actionable way  

---

## Author
**Olise Ebinum**  
Aspiring Data Scientist | Machine Learning Enthusiast  
GitHub: https://github.com/olisebinum  
Email: olisebinum@gmail.com  

---

*“Great models don’t just predict—they reveal the relationships that shape real-world outcomes.”*

