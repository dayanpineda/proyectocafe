
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

# --- Data Summary ---
# Save df.info() output
with open('df_info.txt', 'w') as f:
    df.info(buf=f)

# Save df.describe() output
df.describe().to_csv('df_describe.csv')


# --- Visualizations ---

# 1. Attrition Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Attrition', data=df)
plt.title('Distribución de Attrition')
plt.savefig('plots/attrition_distribution.png')
plt.close()

# 2. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribución de Age')
plt.xlabel('Age')
plt.ylabel('Frecuencia')
plt.savefig('plots/age_distribution.png')
plt.close()

# 3. Job Satisfaction vs. Attrition
plt.figure(figsize=(10, 6))
sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, palette='viridis')
plt.title('Satisfacción Laboral vs. Renuncia')
plt.xlabel('Nivel de Satisfacción Laboral')
plt.ylabel('Cantidad de Empleados')
plt.legend(title='Renuncia')
plt.tight_layout()
plt.savefig('plots/satisfaction_vs_attrition.png')
plt.close()

# 4. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.title('Distribución por Gender')
plt.savefig('plots/gender_distribution.png')
plt.close()

# 5. Monthly Income Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['MonthlyIncome'], kde=True, bins=30)
plt.title('Distribución de MonthlyIncome')
plt.xlabel('Monthly Income')
plt.ylabel('Frecuencia')
plt.savefig('plots/monthly_income_distribution.png')
plt.close()


print("EDA script finished. Plots and data summaries are saved.")

# --- Key Metrics ---
total_employees = len(df)
attrition_count = df['Attrition'].value_counts().get('Yes', 0)
attrition_rate = (attrition_count / total_employees) * 100 if total_employees > 0 else 0
average_age = df['Age'].mean()
average_income = df['MonthlyIncome'].mean()

print("\n--- Key Metrics ---")
print(f"Total Employees: {total_employees}")
print(f"Attrition Rate: {attrition_rate:.2f}%")
print(f"Average Age: {average_age:.2f}")
print(f"Average Monthly Income: ${average_income:,.2f}")

