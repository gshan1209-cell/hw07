# Boston Housing Regression Report

## 1. Introduction

This project solves the Boston Housing regression problem. Given multiple housing and neighborhood-related features, the system builds machine learning regression models to predict the median value of owner-occupied homes (MEDV).

## 2. Dataset Description

The Boston Housing dataset contains 506 samples with 13 features and 1 target variable (MEDV).

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for large lots |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable |
| NOX | Nitric oxides concentration |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built before 1940 |
| DIS | Weighted distances to employment centers |
| RAD | Index of accessibility to radial highways |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio by town |
| B | Legacy demographic variable |
| LSTAT | Percentage of lower status population |
| MEDV | Target: median value of owner-occupied homes ($1000s) |

## 3. Problem Definition

The target variable is MEDV, which represents the median value of owner-occupied homes in thousands of dollars. The problem is a supervised regression problem.

## 4. Data Preprocessing

- Data split: 80% training (404 samples), 20% testing (102 samples)
- Random state: 42 for reproducibility
- Features standardized using StandardScaler before model training

## 5. Exploratory Data Analysis

The dataset was explored through:
- Correlation analysis between features and target
- Feature distribution visualization
- Missing value checking (no missing values found)

## 6. Feature Selection

9 feature selection methods were applied:

1. **Pearson Correlation**: Linear correlation between each feature and MEDV
2. **Spearman Correlation**: Rank-based correlation
3. **F-test Regression**: Univariate F-test scores
4. **Mutual Information**: Information gain between features and target
5. **RFE**: Recursive Feature Elimination with LinearRegression
6. **SFS (Forward)**: Sequential Forward Selection
7. **SBS (Backward)**: Sequential Backward Selection
8. **Lasso (L1)**: L1 regularization coefficient ranking
9. **Random Forest**: Feature importance from ensemble trees

## 7. Model Training

LinearRegression was used as the default evaluation model to fairly compare feature selection methods. Each method's top-k features (k=1 to 13) were evaluated separately.

## 8. Model Evaluation

Evaluation metrics:
- **R²** (Coefficient of Determination): Higher is better
- **MSE** (Mean Squared Error): Lower is better

### Key Findings

- Best single feature: LSTAT (R² ≈ 0.54)
- Strongest feature pair: LSTAT + RM (R² ≈ 0.57)
- Sweet spot at k=3: LSTAT + RM + PTRATIO (R² varies by method)
- Most methods converge around k=7-8 features
- RM and LSTAT consistently rank as the most important features

## 9. Result Visualization

![Feature Selection Performance Comparison](reports/figures/feature_selection_performance_allinone.png)

Figure: Feature selection and model performance comparison for the Boston Housing regression problem.

## 10. Discussion

The analysis shows that only a few features are needed to achieve reasonable prediction performance. LSTAT (% lower status population) and RM (average rooms per dwelling) consistently emerge as the most important predictors across all feature selection methods. Adding more than 7-8 features yields diminishing returns in R² improvement.

Different feature selection methods produce slightly different rankings, particularly for mid-ranked features. Correlation-based methods (Pearson, Spearman) and statistical methods (F-test, Mutual Info) tend to agree on top features, while model-based methods (Lasso, RF) and wrapper methods (RFE, SFS, SBS) may prioritize different combinations.

## 11. Ethical Note

The Boston Housing dataset is commonly used for regression practice, but some variables contain ethically sensitive or outdated social assumptions. This project uses the dataset only for educational purposes. For modern real-world housing applications, more appropriate datasets such as Ames Housing or California Housing are recommended.

## 12. Conclusion

This project demonstrates a complete feature selection and model evaluation pipeline for the Boston Housing regression problem. The all-in-one chart provides a visual summary of how different feature selection methods and feature subset sizes affect prediction performance. The key insight is that a small number of well-chosen features can achieve performance close to using all features, which has practical implications for model simplicity and interpretability.
