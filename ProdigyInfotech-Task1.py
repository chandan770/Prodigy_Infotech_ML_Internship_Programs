import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("train.csv")

print(df.head())
print(df.shape)

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

data = df[features + [target]].copy()

data = data.dropna()

X = data[features]
y = data[target]

print(data.describe())

plt.figure(figsize=(8, 5))
plt.scatter(data["GrLivArea"], data["SalePrice"], alpha=0.5)
plt.xlabel("Living Area (Square Feet)")
plt.ylabel("Sale Price")
plt.title("Living Area vs House Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=data["BedroomAbvGr"], y=data["SalePrice"])
plt.xlabel("Number of Bedrooms")
plt.ylabel("Sale Price")
plt.title("Bedrooms vs House Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=data["FullBath"], y=data["SalePrice"])
plt.xlabel("Number of Bathrooms")
plt.ylabel("Sale Price")
plt.title("Bathrooms vs House Price")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(comparison.head(10))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

print("Intercept:", model.intercept_)

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print(coefficients)

area = float(input("Enter house area in square feet: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

new_house = pd.DataFrame({
    "GrLivArea": [area],
    "BedroomAbvGr": [bedrooms],
    "FullBath": [bathrooms]
})

predicted_price = model.predict(new_house)

print("Predicted House Price: $", round(predicted_price[0], 2))