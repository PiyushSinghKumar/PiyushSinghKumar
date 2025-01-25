# Statistics.ipynb

## 📌 Overview

This Jupyter Notebook contains statistical analyses on two datasets using ANOVA and Principal Component Analysis (PCA). The analysis is structured into two problem statements, each focusing on different statistical techniques.

## 🏥 Problem 1: ANOVA on Hay Fever Compound Study

A research laboratory is testing a new compound for hay fever relief. The effects of two active ingredients, A & B, were analyzed using ANOVA techniques.

### 🔍 Analysis Steps:
1. **State the Hypothesis:**
   - Formulate Null and Alternative Hypotheses for One-Way ANOVA on A & B.
2. **One-Way ANOVA:**
   - Perform One-Way ANOVA on A & B separately against the relief time.
   - Interpret whether the Null Hypothesis is accepted or rejected.
3. **Interaction Analysis:**
   - Use an interaction plot to visualize the relationship between A & B.
4. **Two-Way ANOVA:**
   - Conduct Two-Way ANOVA to analyze the effects of A, B, and their interaction on relief time.
5. **Business Implications:**
   - Discuss the relevance of ANOVA findings in a pharmaceutical setting.

Dataset: `Fever.csv`

---

## 🎓 Problem 2: Principal Component Analysis on Education Dataset

This case study examines various institutions based on multiple parameters. PCA is applied to reduce dimensionality and identify important factors affecting the institutions' ratings.

### 🔍 Analysis Steps:
1. **Exploratory Data Analysis (EDA):**
   - Conduct univariate and multivariate analysis.
2. **Scaling:**
   - Scale the data and justify the chosen scaling technique.
3. **Covariance vs Correlation Matrix:**
   - Compare and contrast both matrices.
4. **Outlier Detection:**
   - Identify and analyze outliers before and after scaling.
5. **PCA Implementation:**
   - Compute covariance matrix, eigenvalues, and eigenvectors.
   - Extract the first Principal Component (PC).
   - Analyze cumulative eigenvalues for optimal PC selection.
6. **Business Implications:**
   - Interpret the significance of PCA results in the education sector.
