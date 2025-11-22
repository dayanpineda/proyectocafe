
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Asegurarse de que el directorio de gráficos exista
if not os.path.exists('plots'):
    os.makedirs('plots')

# Cargar el dataset
file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

# Seleccionar solo las columnas numéricas para el análisis de correlación
df_numeric = df.select_dtypes(include=np.number)

# Eliminar columnas sin señal que ya identificamos
# Aunque EmployeeNumber es numérico, es un ID y no debe estar en el análisis de correlación.
cols_to_drop = ['EmployeeCount', 'StandardHours', 'EmployeeNumber']
df_numeric = df_numeric.drop(columns=cols_to_drop, errors='ignore')

# Calcular la matriz de correlación
corr_matrix = df_numeric.corr()

# Configurar el gráfico
plt.figure(figsize=(20, 18))

# Crear una máscara para ocultar la parte superior del mapa de calor (espejada)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Generar el mapa de calor
sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='viridis', 
            linewidths=.5,
            cbar_kws={"shrink": .8})

plt.title('Mapa de Calor de Correlación de Variables Numéricas', fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar la figura
plot_filename = 'plots/correlation_heatmap.png'
plt.savefig(plot_filename, dpi=150)
plt.close()

print(f"Se generó y guardó el mapa de calor de correlación en {plot_filename}")
