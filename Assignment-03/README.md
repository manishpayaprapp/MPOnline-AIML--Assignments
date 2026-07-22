# AI-ML Assignment 3

## Student Details
- **Name:** Manish Satish Payaprapp
- **Registration No:** 23BCY10046
- **Application No:** IN26009666

## Objective
Predict employee salary using Polynomial Regression.

## Dataset Link
https://www.kaggle.com/datasets/akram24/position-salaries

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn

## Methodology
1. Load dataset
2. Preprocess data
3. Split train/test (80:20)
4. Polynomial Features (degree=3)
5. Train Linear Regression on transformed data
6. Evaluate using MAE, MSE, R²
7. Plot regression curve

## Results
Displays MAE, MSE and R² after prediction.

## Conclusion
Polynomial Regression models non-linear relationships by adding higher-degree terms. It performs better than simple Linear Regression on datasets where the relationship between variables is curved. In the Position Salaries dataset, salary increases non-linearly with position level, making Polynomial Regression a better choice. It provides a closer fit to the data and generally improves prediction accuracy while remaining easy to implement.
