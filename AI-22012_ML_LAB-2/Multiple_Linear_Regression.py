import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('Real-estate.csv')

X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

print("Features (X):")
print(X)
print("\nTarget (y):")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

Multiple_linear_regressor = LinearRegression()
Multiple_linear_regressor.fit(X_train, y_train)

Y_pred = Multiple_linear_regressor.predict(X_test)

print("\nCoefficients:", Multiple_linear_regressor.coef_)
print("\nIntercept:", Multiple_linear_regressor.intercept_)
print('\nVariance score: {}'.format(Multiple_linear_regressor.score(X_test, y_test)))
print('\nMAE:', metrics.mean_absolute_error(y_test, Y_pred))
print('\nMSE:', metrics.mean_squared_error(y_test, Y_pred))
print('\nRMSE:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))

input_data = [[415, 2012.500, 4.5, 95.45606, 6, 29.97433, 127.5431]]
predicted_price = Multiple_linear_regressor.predict(input_data)
print("\nPredicted House Price for separate input data:", predicted_price)


plt.style.use('fivethirtyeight')
plt.scatter(Multiple_linear_regressor.predict(X_train), Multiple_linear_regressor.predict(X_train) - y_train, color="green", s=10, label='Train data')
plt.scatter(Multiple_linear_regressor.predict(X_test), Multiple_linear_regressor.predict(X_test) - y_test, color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=min(y), xmax=max(y), linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.xlabel("Predicted House Price")
plt.ylabel("Error")
plt.show()

residuals = y_test - Y_pred

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(residuals, kde=True, color='black', alpha=0.5, ax=ax1)
ax1.set_title('Histogram Plot of Residuals for Multiple Linear Regression')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Frequency')
ax1.grid(True, ls='--', alpha=0.5)

ax2.scatter(Y_pred, y_test, color='blue', s=10)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_title('Predicted vs Actual')
ax2.set_xlabel('Predicted House Price')
ax2.set_ylabel('Actual House Price')
ax2.grid(True, ls='--', alpha=0.5)

ax3.scatter(Y_pred, residuals, color='green', s=10)
ax3.axhline(0, color='red', linestyle='--')
ax3.set_title('Residuals vs Predicted')
ax3.set_xlabel('Predicted House Price')
ax3.set_ylabel('Residuals')
ax3.grid(True, ls='--', alpha=0.5)

ax4.axis('off')

plt.tight_layout()
plt.show()

