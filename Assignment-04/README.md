# Assignment 4 – Breast Cancer Classification using K-Nearest Neighbors (KNN)

### **Name:** Manish Satish Payaprapp
### **Reg.No:** 23BCY10046
### **Application.No:** IN26009666

## Objective
Build a K-Nearest Neighbors (KNN) classification model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements from digitized images of breast mass cell nuclei.

## Dataset Link
Breast Cancer Wisconsin (Diagnostic) Data Set — Kaggle:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> Note: The dataset is **not included in this repository**. Download it from the Kaggle link above and place it as `data.csv` in the project root before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (`train_test_split`, `StandardScaler`, `KNeighborsClassifier`, evaluation metrics)

## Methodology
1. **Data Understanding** – Loaded the dataset, inspected the first five records, identified numerical features and the target variable (`diagnosis`), and reviewed dataset info and summary statistics.
2. **Data Preprocessing** – Checked for missing values, dropped the non-predictive `id` column, label-encoded the target (M = 1, B = 0), standardized all feature values using `StandardScaler`, and split the data into 80% training / 20% testing sets.
3. **Model Development** – Trained a `KNeighborsClassifier` with **K = 5** on the standardized training data and generated predictions on the test set.
4. **Model Evaluation** – Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and visualized performance with a confusion matrix. Also compared accuracy across K values from 1 to 20.

## Results
| Metric | Score |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9744 |
| Recall | 0.9048 |
| F1-Score | 0.9383 |

The confusion matrix and accuracy-vs-K plot are generated inside the notebook (`Assignment-4.ipynb`) and saved as `confusion_matrix.png` and `accuracy_vs_k.png`.

**Key observations:**
- The model correctly classifies the large majority of tumors, with very few misclassifications.
- Precision is high, meaning that when the model predicts "Malignant," it is very likely correct — important for minimizing unnecessary alarm.
- Recall is slightly lower than precision, meaning a small number of malignant cases are misclassified as benign (false negatives), which is the more clinically risky error type and would need closer attention in a real deployment.
- Model accuracy remains fairly stable across a range of K values once features are standardized.

## Conclusion
The KNN model achieved strong performance on the Breast Cancer Wisconsin (Diagnostic) dataset, confirming that the diagnostic measurements carry strong discriminative power between malignant and benign tumors. Feature scaling was essential, since KNN relies on Euclidean distance and unscaled features (e.g., area vs. smoothness) would otherwise bias the distance calculation. A key limitation of KNN is its computational cost at prediction time, since it must compute distances to every training point for each new prediction, making it less scalable to very large datasets; it is also sensitive to noisy/irrelevant features and the chosen value of K.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace Assignment-4.ipynb
```
