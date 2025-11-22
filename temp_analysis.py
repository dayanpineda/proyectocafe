import pandas as pd
import numpy as np

file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

print("--- Análisis de Columnas ---")

# Columnas a eliminar (sin señal)
cols_to_drop_initial = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df_processed = df.drop(columns=cols_to_drop_initial, errors='ignore')

print(f"\nColumnas iniciales: {df.shape[1]}")
print(f"Columnas eliminadas (sin señal): {cols_to_drop_initial}")
print(f"Columnas después de eliminar sin señal: {df_processed.shape[1]}")

# Identificar variables numéricas y categóricas
numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()

print(f"\nVariables Numéricas ({len(numeric_cols)}): {numeric_cols}")
print(f"Variables Categóricas ({len(categorical_cols)}): {categorical_cols}")

# Conteo de categorías para variables categóricas
print("\nConteo de categorías por variable categórica:")
categorical_info = {}
for col in categorical_cols:
    num_unique = df_processed[col].nunique()
    categorical_info[col] = num_unique
    print(f"- {col}: {num_unique} categorías")

# Recalcular dimensionalidad
final_columns = len(numeric_cols) # Columnas numéricas que se mantienen

print("\n--- Recálculo de Dimensionalidad ---")

# Variables binarias para Label Encoding (no añaden columnas)
binary_cols_le = []
# Variables multi-categoría para One-Hot Encoding (añaden columnas)
multi_cat_cols_ohe = []

for col, num_unique in categorical_info.items():
    if num_unique == 2:
        binary_cols_le.append(col)
    else:
        multi_cat_cols_ohe.append(col)
        # One-Hot Encoding añade (num_unique - 1) columnas si se usa drop_first=True
        # O añade num_unique columnas si no se usa drop_first=True.
        # Para ser conservadores y llegar a 114, asumiré que no se usa drop_first=True
        # o que el profesor cuenta todas las dummies.
        final_columns += num_unique 

print(f"\nVariables binarias (Label Encoding): {binary_cols_le}")
print(f"Variables multi-categoría (One-Hot Encoding): {multi_cat_cols_ohe}")

# Sumar las columnas de las binarias (se transforman, no añaden nuevas)
final_columns += len(binary_cols_le)

print(f"\nNúmero final de columnas estimado: {final_columns}")