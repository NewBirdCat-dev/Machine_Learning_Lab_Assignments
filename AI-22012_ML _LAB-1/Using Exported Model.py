import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

model = joblib.load('linear_regression_model.pkl')

housing = pd.read_csv('housing.csv')
X = housing[['median_income']]
y = housing['median_house_value']

X_feature_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_feature_scaled, y, test_size=0.2, random_state=42)

predictions = model.predict(X_test)

score_r2_test = model.score(X_test, y_test)
score_error_test = mean_squared_error(y_test, predictions, squared=False)

print(f"R² Test = {score_r2_test:.4f}, RMSE = {score_error_test:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test.flatten(), y=y_test, color='blue', label='Data Points')
sns.regplot(x=X_test.flatten(), y=predictions, color='red', label='Regression Line', scatter=False)
plt.title("Univariate Linear Regression on California Housing Dataset")
plt.xlabel("Median Income (scaled)")
plt.ylabel("Target Variable (Median House Value)")
plt.legend()
plt.show()
