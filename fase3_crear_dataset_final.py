
import pandas as pd
import numpy as np

# Cargar el dataset original
file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

# --- 1. Eliminación de Variables sin Señal ---
cols_to_drop_initial = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df_processed = df.drop(columns=cols_to_drop_initial, errors='ignore')

# --- 2. Eliminación de Variables por Colinealidad ---
# Basado en el análisis previo, eliminamos 'JobLevel', 'YearsInCurrentRole', 'YearsWithCurrManager'
cols_to_drop_collinearity = ['JobLevel', 'YearsInCurrentRole', 'YearsWithCurrManager']
df_processed = df_processed.drop(columns=cols_to_drop_collinearity, errors='ignore')

# --- 3. Codificación de Variables (para llegar a 114 columnas) ---
# Identificar las columnas que se mantendrán como numéricas continuas
# (aquellas con un número muy alto de valores únicos)
continuous_cols = ['DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate']

# Todas las demás columnas se tratarán como categóricas para One-Hot Encoding
# (incluyendo 'object' y las numéricas discretas/ordinales)
cols_to_encode = [col for col in df_processed.columns if col not in continuous_cols]

# Aplicar One-Hot Encoding a las columnas seleccionadas
# Usamos pd.get_dummies que maneja tanto 'object' como tipos numéricos
# drop_first=False para generar una columna por cada categoría
df_encoded = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=False, dtype=float)

# El resultado de get_dummies ya contiene las columnas continuas que no se tocaron,
# así que df_encoded es nuestro dataset final.

# --- 4. Guardar el Dataset Procesado ---
output_filename = 'dataset_procesado.csv'
df_encoded.to_csv(output_filename, index=False)

print(f"Dataset procesado y guardado como '{output_filename}'.")
print(f"Número de columnas iniciales (útiles): {df_processed.shape[1]}")
print(f"Número de columnas finales (después de OHE): {df_encoded.shape[1]}")

# Verificación para asegurar que el número de columnas es el esperado
# Columnas continuas: 4
# Columnas a codificar: 31 (total) - 4 (continuas) - 3 (colinealidad) = 24
# Suma de categorías únicas en las 24 columnas a codificar:
total_categories = 0
for col in cols_to_encode:
    if col in df.columns: # Asegurarse de que la columna existe en el df original
        total_categories += df[col].nunique()

final_col_check = len(continuous_cols) + total_categories
print(f"Verificación del cálculo de columnas: {len(continuous_cols)} (continuas) + {total_categories} (categorías) = {final_col_check}")
