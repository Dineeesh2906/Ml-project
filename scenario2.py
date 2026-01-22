
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/diabetescsv/diabetes.csv")

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()


avg_glucose = df.groupby("Outcome")["Glucose"].mean()
print("\nAverage Glucose by Outcome:")
print(avg_glucose)

plt.figure(figsize=(6,4))
avg_glucose.plot(kind="bar")
plt.title("Average Glucose Level by Outcome")
plt.xlabel("Outcome")
plt.ylabel("Average Glucose")
plt.show()

  
plt.figure(figsize=(8,5))
plt.scatter(df["Age"], df["Glucose"], c=df["Outcome"], cmap="coolwarm", alpha=0.7)
plt.title("Age vs Glucose Levels")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.colorbar(label="Outcome")
plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

