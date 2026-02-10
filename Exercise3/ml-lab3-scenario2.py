
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("/kaggle/input/autompgv/auto-mpg.csv")


df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')


X = df[['horsepower']]
y = df['mpg']


X = X.fillna(X.mean())
y = y.fillna(y.mean())

def evaluate(y_test, y_pred, degree):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nDegree {degree}")
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

train_error = []
test_error = []


for degree in [2,3,4]:

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate(y_test, y_pred, degree)

    train_error.append(mean_squared_error(y_train, model.predict(X_train)))
    test_error.append(mean_squared_error(y_test, y_pred))

    # Curve Visualization
    plt.scatter(X['horsepower'], y, color='blue')

    x_range = pd.DataFrame({
        'horsepower': np.linspace(X['horsepower'].min(),
                                 X['horsepower'].max(), 100)
    })

    x_poly = poly.transform(x_range)
    x_poly = scaler.transform(x_poly)

    plt.plot(x_range, model.predict(x_poly), color='red')
    plt.title(f"Polynomial Fit Degree {degree}")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.show()


poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
X_poly = StandardScaler().fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_ridge = ridge.predict(X_test)

evaluate(y_test, y_ridge, "4 + Ridge")


plt.plot([2,3,4], train_error, marker='o', label="Train Error")
plt.plot([2,3,4], test_error, marker='o', label="Test Error")
plt.title("Training vs Testing Error")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.legend()
plt.show()


poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_poly = StandardScaler().fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)

residuals = y_test - model.predict(X_test)

plt.hist(residuals, bins=20)
plt.title("Residual Distribution (Degree 3)")
plt.show()
