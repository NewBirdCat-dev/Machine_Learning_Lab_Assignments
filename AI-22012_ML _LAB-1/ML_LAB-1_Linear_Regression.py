import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('Fish[1].csv')
df = df.iloc[:, 1:]
X = df.drop(columns=['Weight']).values
y = df['Weight'].values

X = scale(X)

lr_r2 = []
lr_error = []

linear_regression = linear_model.LinearRegression()

for i in range(X.shape[1]):
    X_i = X[:, i].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_i, y, test_size=0.2, random_state=42)
    
    linear_regression.fit(X_train, y_train)
    
    score_r2_train = linear_regression.score(X_train, y_train)
    score_r2_test = linear_regression.score(X_test, y_test)
    lr_r2.append((i + 1, score_r2_train, score_r2_test))
    
    score_error_train = mean_squared_error(y_train, linear_regression.predict(X_train), squared=False)
    score_error_test = mean_squared_error(y_test, linear_regression.predict(X_test), squared=False)
    lr_error.append((i + 1, score_error_train, score_error_test))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_i.flatten(), y=y, color='blue', label='Data Points')
    sns.regplot(x=X_test.flatten(), y=linear_regression.predict(X_test), color='red', label='Regression Line', scatter=False)
    plt.title(f"Univariate Linear Regression for Feature {i + 1}")
    plt.xlabel(f"Feature {i + 1}")
    plt.ylabel("Weight of Fish (grams)")
    plt.legend()
    plt.show()

    print(f"Feature {i + 1}: R² Train = {score_r2_train:.4f}, R² Test = {score_r2_test:.4f}")

joblib.dump(linear_regression, 'linear_regression_model.pkl')
