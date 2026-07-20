import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("insurance.csv")

print(df.head())
print(df.info())

print("Numerical Features:", df.select_dtypes(include=["int64","float64"]).columns.tolist())
print("Categorical Features:", df.select_dtypes(include=["object"]).columns.tolist())
print("Target Variable: charges")

print(df.isnull().sum())

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.grid(True)
plt.show()

print("""
Conclusion:
This project predicts medical insurance charges using Multiple Linear Regression.
Smoking status, BMI and age have the greatest influence on insurance charges.
A limitation is that Linear Regression assumes linear relationships and may not
capture nonlinear interactions.
""")
