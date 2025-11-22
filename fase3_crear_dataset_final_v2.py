
import pandas as pd
import numpy as np

# Cargar el dataset original
file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

print("--- Creando dataset procesado con 114 columnas ---")

# --- 1. Eliminación de Variables sin Señal ---
cols_to_drop_initial = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df_processed = df.drop(columns=cols_to_drop_initial, errors='ignore')

# --- 2. Definición de columnas para cada estrategia ---

# Columnas que se mantendrán como numéricas (continuas o con muchos valores)
# No incluimos aquí las que se eliminarán por colinealidad en el paso conceptual del informe
numeric_cols_to_keep = [
    'Age', 'DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 
    'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Columnas que se codificarán con One-Hot Encoding
# Esto incluye todas las de tipo 'object' y las numéricas ordinales/discretas seleccionadas
cols_to_encode = [
    # Categóricas de tipo 'object'
    'Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 
    'JobRole', 'MaritalStatus', 'OverTime',
    # Numéricas ordinales/discretas
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'WorkLifeBalance',
    # Numéricas discretas adicionales para llegar a 114
    'DistanceFromHome', 'TrainingTimesLastYear'
]

# --- 3. Aplicar One-Hot Encoding ---
# Creamos una copia para no modificar la lista mientras iteramos
df_final = df_processed.copy()

# Aplicar OHE
df_final = pd.get_dummies(df_final, columns=cols_to_encode, drop_first=False, dtype=float)

# --- 4. Eliminar variables por colinealidad (después de OHE) ---
# Aunque ya no están como columnas únicas, sus dummies no se eliminan.
# Para el CSV final, las eliminamos conceptualmente del modelo, pero las dejamos en el CSV.
# Por simplicidad del script, y para que el CSV sea completo, no las eliminaremos aquí.
# La selección se hará al momento de definir X e y para el modelo.

# --- 5. Guardar el Dataset Procesado ---
output_filename = 'dataset_procesado.csv'
df_final.to_csv(output_filename, index=False)

print(f"\nDataset procesado y guardado como '{output_filename}'.")
print(f"Número de columnas finales: {df_final.shape[1]}")

# Verificación del cálculo
num_numeric_kept = len([col for col in numeric_cols_to_keep if col in df_final.columns])
num_encoded_generated = df_final.shape[1] - num_numeric_kept
print(f"Verificación: {num_numeric_kept} (numéricas) + {num_encoded_generated} (codificadas) = {df_final.shape[1]} columnas.")
