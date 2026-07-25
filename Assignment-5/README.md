# Assignment 5 – Employee Attrition Prediction using Decision Tree and Random Forest Classification

### **Name:** Manish Satish Payaprapp
### **Reg.No:** 23BCY10046
### **Application.No:** IN26009666

## Objective
Build and compare **Decision Tree** and **Random Forest** classification models to predict whether an employee is likely to leave the organization (**Attrition**), based on demographic, professional, and work-related attributes.

## Dataset Link
IBM HR Analytics Employee Attrition & Performance — Kaggle:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

> Note: The dataset is **not included in this repository**. Download it from the Kaggle link above (file: `WA_Fn-UseC_-HR-Employee-Attrition.csv`) and place it in the project root before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (`train_test_split`, `DecisionTreeClassifier`, `RandomForestClassifier`, evaluation metrics)

## Methodology
1. **Data Understanding** – Loaded the dataset, inspected the first five records, identified numerical features, categorical features, and the target variable (`Attrition`), and reviewed dataset info and summary statistics.
2. **Data Preprocessing** – Checked for missing values, dropped non-informative columns (`EmployeeCount`, `StandardHours`, `Over18` are constant; `EmployeeNumber` is just an ID), label-encoded the target (Yes=1, No=0), one-hot encoded the remaining categorical variables, and split the data into 80% training / 20% testing sets.
3. **Model Development** – Trained a `DecisionTreeClassifier` (Model 1) and a `RandomForestClassifier` with 100 estimators (Model 2) on the same training data, and generated predictions on the same test set for both.
4. **Model Evaluation & Comparison** – Evaluated both models using Accuracy, Precision, Recall, and F1-Score, plotted a confusion matrix for each, and generated a feature importance plot for the Random Forest model.
5. **Bonus (optional)** – Compared the default Random Forest against a version with `max_depth=5` to observe the effect of that hyperparameter.

## Results

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | 0.7653 | 0.8333 |
| Precision | 0.3103 | 0.4167 |
| Recall | 0.3830 | 0.1064 |
| F1-Score | 0.3429 | 0.1695 |

Confusion matrices for both models and the Random Forest feature importance plot are generated inside the notebook (`Assignment-5.ipynb`) and saved as `confusion_matrices.png` and `feature_importance.png`.

## Model Comparison
The two models trade off on different metrics rather than one clearly beating the other:
- **Random Forest** achieves noticeably higher **accuracy and precision**.
- **Decision Tree** achieves higher **recall and F1-score** on the minority "Attrition = Yes" class.

This happens because the dataset is **imbalanced** (most employees do not leave). The default Random Forest leans toward predicting the majority class, which raises overall accuracy/precision but causes it to miss more true attrition cases (lower recall) than the single, more minority-sensitive Decision Tree.

**Which model is "better" depends on the goal:** if minimizing false alarms and maximizing overall correctness matters most, Random Forest is preferable. If the priority is catching as many at-risk employees as possible — usually the real goal in attrition prediction — the Decision Tree's higher recall/F1 makes it more useful in this run, though in practice this gap is normally closed by addressing class imbalance (e.g., `class_weight='balanced'`, SMOTE, or threshold tuning) rather than switching models.

The top predictors identified by the Random Forest's feature importance are `OverTime`, `MonthlyIncome`, `Age`, `TotalWorkingYears`, and job-role/stock-option-related features — consistent with common HR intuition.

## Conclusion
Comparing the two models produced a mixed result: Random Forest won on accuracy and precision, while the Decision Tree won on recall and F1-score for the minority class, due to class imbalance in the dataset. In general, Random Forest tends to outperform a single Decision Tree because it builds many trees on bootstrap samples and random feature subsets, then aggregates their predictions — this ensembling reduces variance and typically produces a more robust model than any single, easily-overfit tree.

A key limitation of Decision Trees is their tendency to **overfit** the training data, especially when grown deep without pruning; they are also unstable, meaning small changes in training data can produce a very different tree.

A key limitation of Random Forest is its **reduced interpretability** compared to a single tree, since it aggregates the votes of many trees, making individual decisions harder to trace. It is also more computationally expensive, and — as seen here — can still under-perform on the minority class of an imbalanced dataset unless that imbalance is explicitly addressed.

## Bonus Challenge (Not Mandatory)
Tested the effect of `max_depth` on the Random Forest by comparing the default (`max_depth=None`) model against `max_depth=5`. The depth-limited forest produced accuracy/precision close to the default, with a slight further drop in recall — indicating the recall gap seen above is driven more by class imbalance than by tree depth/overfitting. `max_depth` generally controls the bias-variance trade-off: smaller values increase bias but reduce overfitting risk, while larger/unrestricted values do the opposite.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace Assignment-5.ipynb
```
