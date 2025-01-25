import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

data = pd.read_csv('50_Startups.csv')

X = data.drop('Profit', axis=1)
y = data['Profit']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

one_hot_columns = ct.get_feature_names_out(input_features=data.columns.drop('Profit').values)
predictor_names = np.concatenate([data.columns.drop('Profit').values[:3], one_hot_columns])

degrees = [2, 3, 4, 5]

for predictor_index in range(X.shape[1]):
    predictor_name = predictor_names[predictor_index]
    print(f"\nAnalyzing predictor: {predictor_name}")

    features = X[:, predictor_index].reshape(-1, 1)

    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(features, y, test_size=0.2, random_state=0)

    models = []
    X_train_transformed = []
    X_test_transformed = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_transformed.append(poly.fit_transform(X_train_poly))
        X_test_transformed.append(poly.transform(X_test_poly))
        model = LinearRegression()
        model.fit(X_train_transformed[-1], y_train_poly)
        models.append(model)

    y_train_preds = [model.predict(X) for model, X in zip(models, X_train_transformed)]
    y_test_preds = [model.predict(X) for model, X in zip(models, X_test_transformed)]

    for i, degree in enumerate(degrees):
        print(f"\nDegree {degree}:")
        print("Coefficients:", models[i].coef_)
        print("Intercept:", models[i].intercept_)
        print('Variance score:', r2_score(y_test_poly, y_test_preds[i]))
        print('MAE:', metrics.mean_absolute_error(y_test_poly, y_test_preds[i]))
        print('MSE:', metrics.mean_squared_error(y_test_poly, y_test_preds[i]))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_poly, y_test_preds[i])))

    plt.style.use('fivethirtyeight')
    for i, degree in enumerate(degrees):
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test_preds[i], y_test_preds[i] - y_test_poly, color="blue", s=10, label='Test data')
        plt.hlines(y=0, xmin=min(y), xmax=max(y), linewidth=2)
        plt.title(f"Residual errors for Degree {degree} with Predictor {predictor_name}")
        plt.xlabel("Predicted Profit")
        plt.ylabel("Error")
        plt.legend(loc='upper right')
        plt.show()

    residuals = [y_test_poly - y_pred for y_pred in y_test_preds]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, degree in enumerate(degrees):
        sns.histplot(residuals[i], kde=True, color='black', alpha=0.5, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'Histogram Plot of Residuals for Degree {degree} with Predictor {predictor_name}')
        axes[i//2, i%2].set_xlabel('Residuals')
        axes[i//2, i%2].set_ylabel('Frequency')
        axes[i//2, i%2].grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, degree in enumerate(degrees):
        stats.probplot(residuals[i], dist="norm", plot=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'Q-Q Plot of Residuals for Degree {degree} with Predictor {predictor_name}')
        axes[i//2, i%2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(8, 10))
    for i, degree in enumerate(degrees):
        sns.boxplot(y=residuals[i], color='black', ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'Box Plot of Residuals for Degree {degree} with Predictor {predictor_name}')
        axes[i//2, i%2].set_ylabel('Residuals')
        axes[i//2, i%2].grid(True)

    plt.tight_layout()
    plt.show()

    for i, degree in enumerate(degrees):
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test_poly, y_test_preds[i], color='blue', s=10, label='Predicted vs Actual')
        plt.plot([min(y_test_poly), max(y_test_poly)], [min(y_test_poly), max(y_test_poly)], color='red', linestyle='--', label='Perfect Prediction')
        plt.title(f'Actual vs. Predicted Profit for Degree {degree} with Predictor {predictor_name}')
        plt.xlabel('Actual Profit')
        plt.ylabel('Predicted Profit')
        plt.legend()
        plt.show()

    mse_values = [metrics.mean_squared_error(y_test_poly, y_test_preds[i]) for i in range(len(degrees))]
    r2_values = [r2_score(y_test_poly, y_test_preds[i]) for i in range(len(degrees))]

    plt.figure(figsize=(10, 5))
    plt.plot(degrees, mse_values, marker='o', label='MSE', color='blue')
    plt.plot(degrees, r2_values, marker='o', label='R²', color='orange')
    plt.title(f'Model Complexity vs. Performance for Predictor {predictor_name}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Performance Metrics')
    plt.xticks(degrees)
    plt.legend()
    plt.grid()
    plt.show()

