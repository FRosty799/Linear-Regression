import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
data = pd.read_csv('Student_Marks.csv')

# --- Heatmap Korelasi ---
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# --- Scatter Plot + OLS Regression Line ---
plt.figure(figsize=(8, 5))
sns.scatterplot(x='time_study', y='Marks', data=data, color='black', label='Data Points')

# Menambahkan garis OLS secara manual
X = data['time_study']
y = data['Marks']
X_ols = sm.add_constant(X)  # tambahkan intercept (konstanta)

model = sm.OLS(y, X_ols).fit()
predictions = model.predict(X_ols)

# Garis regresi
plt.plot(X, predictions, color='red', label='OLS Regression Line')
plt.xlabel('Time Study (hours)')
plt.ylabel('Marks')
plt.title('Scatter Plot with OLS Regression Line')
plt.legend()
plt.show()

# --- Ringkasan model OLS ---
print(model.summary())
