import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('50_Startups.csv')

X = df.drop('Profit', axis=1)
y = df['Profit']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

subsets = [
    ['R&D Spend', 'Administration', 'Marketing Spend', 'State'],
    ['R&D Spend', 'Administration', 'State'],
    ['R&D Spend', 'Marketing Spend', 'State'],
    ['Administration', 'Marketing Spend', 'State'],
    ['R&D Spend', 'State'],
    ['Administration', 'State'],
    ['Marketing Spend', 'State']
]

variance_scores = []
maes = []
mses = []
rmses = []

for subset in subsets:
    X_subset = X[:, [i for i, col in enumerate(df.columns) if col in subset]]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=0)

    Multiple_linear_regressor = LinearRegression()
    Multiple_linear_regressor.fit(X_train, y_train)

    Y_pred = Multiple_linear_regressor.predict(X_test)

    variance_score = Multiple_linear_regressor.score(X_test, y_test)
    mae = metrics.mean_absolute_error(y_test, Y_pred)
    mse = metrics.mean_squared_error(y_test, Y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, Y_pred))

    variance_scores.append(variance_score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)

    print(f"Subset: {subset}")
    print(f"Variance score: {variance_score}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print()

print("Performance metrics for each subset:")
for i, subset in enumerate(subsets):
    print(f"Subset: {subset}")
    print(f"Variance score: {variance_scores[i]}")
    print(f"MAE: {maes[i]}")
    print(f"MSE: {mses[i]}")
    print(f"RMSE: {rmses[i]}")
    print()

print("Ranking of subsets based on variance score:")
for i, (subset, variance_score) in enumerate(sorted(zip(subsets, variance_scores), key=lambda x: x[1], reverse=True)):
    print(f"Rank {i+1}: Subset {subset} with variance score {variance_score}")
