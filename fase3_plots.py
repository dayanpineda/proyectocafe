
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Asegurarse de que el directorio de gráficos exista
if not os.path.exists('plots'):
    os.makedirs('plots')

# Cargar el dataset
file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

# --- Gráficos para Análisis de Simetría y Datos Atípicos ---

# Lista de variables numéricas para analizar
numeric_vars_to_plot = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'Age']

for var in numeric_vars_to_plot:
    # Crear figura con 2 subplots: Boxplot e Histograma/KDE
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Análisis de Distribución para {var}', fontsize=16)

    # Boxplot
    sns.boxplot(x=df[var], ax=axes[0], color='#8e2de2')
    axes[0].set_title(f'Diagrama de Caja (Boxplot) de {var}')
    axes[0].set_xlabel('')

    # Histograma y KDE
    sns.histplot(df[var], kde=True, ax=axes[1], color='#4a00e0')
    axes[1].set_title(f'Histograma y Curva de Densidad (KDE) de {var}')
    axes[1].set_xlabel('Valor')
    axes[1].set_ylabel('Frecuencia')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar la figura
    plot_filename = f'plots/distribution_analysis_{var}.png'
    plt.savefig(plot_filename)
    plt.close(fig)

print(f"Se generaron y guardaron {len(numeric_vars_to_plot)} gráficos de distribución.")
