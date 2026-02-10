import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
df = pd.read_csv("/kaggle/input/studentperformance/StudentsPerformance.csv")
print(df.head())
df["final_score"] = (df["math score"] + 
                     df["reading score"] + 
                     df["writing score"]) / 3
encoder = LabelEncoder()

df["parental level of education"] = encoder.fit_transform(df["parental level of education"])
df["test preparation course"] = encoder.fit_transform(df["test preparation course"])
df["gender"] = encoder.fit_transform(df["gender"])
df["lunch"] = encoder.fit_transform(df["lunch"])
X = df[[
    "parental level of education",
    "test preparation course",
    "gender",
    "lunch"
]]

y = df["final_score"]
X = X.fillna(X.mean())
y = y.fillna(y.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coef_df)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge R2:", ridge.score(X_test, y_test))
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Lasso R2:", lasso.score(X_test, y_test))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted")
plt.show()
sns.barplot(x="Coefficient", y="Feature", data=coef_df)
plt.title("Feature Importance")
plt.show()
residuals = y_test - y_pred

sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
