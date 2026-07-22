import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Position_Salaries.csv")
print(df.head())
print(df.info())
print(df.describe())

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

print("Missing values:")
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

poly=PolynomialFeatures(degree=3)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

model=LinearRegression()
model.fit(X_train_poly,y_train)

y_pred=model.predict(X_test_poly)

print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("R2:",r2_score(y_test,y_pred))

X_grid=np.arange(min(X),max(X),0.1).reshape(-1,1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,model.predict(poly.transform(X_grid)))
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

print("Observations:")
print("1. Polynomial regression captures non-linear trends.")
print("2. Degree-3 model fits the salary curve better than a straight line.")
