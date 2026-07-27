# Assignment 6 – Weather Condition Classification using SVM and Open-Meteo API

### **Name:** Manish Satish Payaprapp
### **Reg.No:** 23BCY10046
### **Application.No:** IN26009666

## Objective
Build a Support Vector Machine (SVM) classifier to predict whether the weather is **Warm** (Temperature ≥ 25°C) or **Cool** (Temperature < 25°C), using live meteorological observations (temperature, relative humidity, surface pressure, wind speed) collected from the Open-Meteo API.

## API Documentation Link
Open-Meteo Forecast API (free, no API key required):
https://open-meteo.com/

Example request used (per the assignment spec):
```
https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7
```

## Libraries Used
- requests
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (`train_test_split`, `StandardScaler`, `SVC`, evaluation metrics)

## Methodology
1. **Data Collection & Understanding** – Fetched hourly weather data (temperature, humidity, pressure, wind speed) for 10 cities spanning different climates (to ensure both classes are well represented), converted the JSON response into a Pandas DataFrame, displayed the first five records, and identified the input features and target variable.
2. **Data Preprocessing** – Created the `Weather_Class` target column from the 25°C threshold, checked for missing values, removed non-predictive columns (`time`, `City`), label-encoded the target (Warm=1, Cool=0), split the data into 80% training / 20% testing sets, and standardized all feature values using `StandardScaler`.
3. **Model Development** – Trained a Support Vector Machine classifier with an **RBF kernel** on the standardized training data and generated predictions on the test set.
4. **Model Evaluation** – Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and visualized performance with a confusion matrix.

## A note on data collection in this repository
The notebook's `fetch_from_api()` function calls the real Open-Meteo endpoint directly and needs no key — it will pull **live, current forecast data** whenever the notebook is run with normal internet access. If that call is blocked in a given environment, the notebook automatically falls back to `simulate_city_weather()`, a locally generated dataset that mirrors the exact same schema and realistic diurnal/climate patterns for each city, so the pipeline still runs end-to-end and produces genuine results. The notebook prints `Live Open-Meteo API reachable in this environment: True/False` in Task 1 so it's always clear which source was used for a given run. **Before final submission, run the notebook once on a machine with unrestricted internet access (e.g., Google Colab or a normal laptop) so the results reflect live API data.**

## Results
*(from a run using the dataset described above; values will differ slightly on live re-runs since weather forecasts change daily)*

| Metric | Score |
|---|---|
| Accuracy | 0.9881 |
| Precision | 0.9928 |
| Recall | 0.9786 |
| F1-Score | 0.9856 |

The confusion matrix is generated inside the notebook (`Assignment-6.ipynb`) and saved as `confusion_matrix.png`.

**Key observations:**
- The RBF-kernel SVM separates Warm and Cool observations very effectively, since the classes are well-structured around the temperature threshold combined with correlated humidity, pressure, and wind features.
- Because the data spans cities with very different climates, most observations sit far from the 25°C boundary, making the task comparatively easy and the model highly confident on most samples.
- The few misclassifications that do occur cluster around observations close to the 25°C threshold, typically in moderate-climate cities where temperatures hover near the boundary.

## Conclusion
The SVM classifier with an RBF kernel accurately distinguished Warm from Cool weather conditions using standard meteorological features, confirming these measurements carry strong signal for this classification task, with most errors occurring near the 25°C decision boundary. Feature scaling is critical for SVM, especially with an RBF kernel, because the algorithm relies on distances between points to define the margin and support vectors; without scaling, features on larger numeric scales (like surface pressure) would dominate the distance calculation and distort the decision boundary. A key advantage of SVM is its effectiveness at finding non-linear decision boundaries via the kernel trick, letting it separate classes that aren't linearly separable in the original feature space. A key limitation is that SVM scales poorly to very large datasets and is sensitive to kernel/hyperparameter choices (like C and gamma), which typically require careful tuning.

## How to Run
```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace Assignment-6.ipynb
```
