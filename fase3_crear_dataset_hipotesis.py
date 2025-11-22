import pandas as pd

# Cargar el dataset original
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition (1).csv')

# 1. Seleccionar las variables de la hipótesis inicial + la variable objetivo
variables_hipotesis = [
    'Age',
    'MonthlyIncome',
    'BusinessTravel',
    'JobSatisfaction',
    'OverTime',
    'Attrition'
]
df_hipotesis = df[variables_hipotesis].copy()

# 2. Codificación de variables categóricas
# Label Encoding para variables binarias
df_hipotesis['OverTime'] = df_hipotesis['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df_hipotesis['Attrition'] = df_hipotesis['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# One-Hot Encoding para 'BusinessTravel'
df_hipotesis = pd.get_dummies(df_hipotesis, columns=['BusinessTravel'], drop_first=True, prefix='BusinessTravel')

# Renombrar columnas para mayor claridad
df_hipotesis.rename(columns={
    'BusinessTravel_Travel_Frequently': 'BusinessTravel_Viaja_Frecuentemente',
    'BusinessTravel_Travel_Rarely': 'BusinessTravel_Viaja_Poco'
}, inplace=True)

# 3. Guardar el nuevo dataset procesado
df_hipotesis.to_csv('dataset_hipotesis.csv', index=False)

print("Dataset 'dataset_hipotesis.csv' creado exitosamente.")
print("Columnas del nuevo dataset:")
print(df_hipotesis.columns.tolist())
print(f"El dataset tiene {df_hipotesis.shape[0]} filas y {df_hipotesis.shape[1]} columnas.")
