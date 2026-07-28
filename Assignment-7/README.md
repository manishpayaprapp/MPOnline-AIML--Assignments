# Assignment 7 – Customer Segmentation using K-Means Clustering and PCA

### **Name:** Manish Satish Payaprapp
### **Reg.No:** 23BCY10046
### **Application.No:** IN26009666

## Objective
Segment mall customers into distinct groups based on their annual income and spending behavior using **K-Means Clustering**, and visualize those clusters in two dimensions using **Principal Component Analysis (PCA)**, to support targeted marketing campaigns.

## Dataset Link
Mall Customer Segmentation Dataset — Kaggle:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

> Note: The dataset is **not included in this repository**. Download it from the Kaggle link above (file: `Mall_Customers.csv`) and place it in the project root before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (`StandardScaler`, `KMeans`, `PCA`)

## Methodology
1. **Data Understanding** – Loaded the dataset, inspected the first five records, identified numerical features (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`) and the categorical feature (`Genre`), and reviewed dataset info and summary statistics. As this is an unsupervised task, there is no target variable.
2. **Data Preprocessing** – Checked for missing values, dropped the non-predictive `CustomerID` column, label-encoded `Genre`, and standardized `Annual Income` and `Spending Score` (the two features used for segmentation, per the problem statement) using `StandardScaler`.
3. **Model Development** – Used the **Elbow Method** to determine the optimal number of clusters (K), trained a `KMeans` model with the selected K, assigned a cluster label to every customer, and applied **PCA** to reduce the standardized features to 2 principal components for visualization.
4. **Visualization & Evaluation** – Plotted the elbow curve, a scatter plot of customer clusters (Annual Income vs. Spending Score) with centroids, and a PCA-space scatter plot colored by cluster.

## Results

**Optimal K (Elbow Method): 5** — inertia drops sharply up to K=5, after which the curve flattens noticeably.

**Cluster profile (mean values):**

| Cluster | Age | Annual Income (k$) | Spending Score | Count | Segment description |
|---|---|---|---|---|---|
| 0 | 42.7 | 55.3 | 49.5 | 81 | Average income, average spending (largest group) |
| 1 | 32.7 | 86.5 | 82.1 | 39 | High income, high spending — premium target segment |
| 2 | 25.3 | 25.7 | 79.4 | 22 | Low income, high spending — engaged despite budget |
| 3 | 41.1 | 88.2 | 17.1 | 35 | High income, low spending — cautious high earners |
| 4 | 45.2 | 26.3 | 20.9 | 23 | Low income, low spending — price-conscious, low engagement |

PCA explained variance: **~50.5% (PC1) and ~49.5% (PC2)**, together capturing 100% of the variance (expected here since PCA was applied to exactly 2 standardized input features).

The elbow curve, cluster scatter plot, and PCA visualization are generated inside the notebook (`Assignment-7.ipynb`) and saved as `elbow_curve.png`, `cluster_scatter.png`, and `pca_clusters.png`.

**Key observations:**
1. **Optimal number of clusters:** The elbow curve bends clearly at K=5, matching the well-known natural structure of this dataset.
2. **How PCA helps visualize high-dimensional data:** PCA re-projects data onto the directions of maximum variance, compressing multiple (often correlated) features into 2 dimensions that can be plotted directly. This becomes essential once more than 2 features (e.g., Age, Genre) are included in clustering, where direct visualization is otherwise impossible.
3. **Characteristics of the customer groups:** Five interpretable segments emerged — a large "average" middle group, a premium high-income/high-spending group, a budget-conscious high-income/low-spending group, a highly-engaged low-income/high-spending group, and a low-income/low-engagement group.
4. Because clustering was performed directly on the same 2 standardized features used for PCA, the cluster scatter plot and PCA visualization look nearly identical, confirming the clusters are genuinely well-separated rather than artifacts of dimensionality reduction.

## Conclusion
K-Means clustering successfully segmented mall customers into five distinct, business-interpretable groups based on annual income and spending score, confirmed by both the elbow method and PCA visualization. These segments enable targeted marketing: for example, premium loyalty offers for high-income high-spenders (Cluster 1), budget-friendly promotions for low-income segments (Clusters 2 and 4), and re-engagement campaigns for high-income customers who currently spend little (Cluster 3).

A key limitation of K-Means clustering is that it requires the number of clusters (K) to be chosen in advance and assumes roughly spherical, similarly-sized clusters, which can misrepresent data with more complex or irregular groupings. A key advantage of PCA is that it reduces high-dimensional data into a small number of components that retain most of the original variance, making it possible to visualize and interpret patterns that would otherwise be impossible to plot directly.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace Assignment-7.ipynb
```
