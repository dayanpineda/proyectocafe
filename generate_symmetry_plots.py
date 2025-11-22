import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
if not os.path.exists('plots_cafe'):
    os.makedirs('plots_cafe')

# Plot styling
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['text.color'] = '#f0f2f5'
plt.rcParams['axes.labelcolor'] = '#f0f2f5'
plt.rcParams['xtick.color'] = '#f0f2f5'
plt.rcParams['ytick.color'] = '#f0f2f5'

print("Loading data...")
df = pd.read_csv('arabica_data_cleaned.csv')

# Variables to analyze for symmetry/outliers
vars_to_plot = ['altitude_mean_meters', 'Moisture', 'Category.One.Defects', 'Category.Two.Defects']

print("Generating symmetry plots...")
for var in vars_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[var].dropna(), kde=True, color='#8e2de2')
    plt.title(f'Distribución de {var}', color='white')
    plt.xlabel(var, color='white')
    plt.ylabel('Frecuencia', color='white')
    plt.savefig(f'plots_cafe/distribution_analysis_{var}.png')
    plt.close()

print("Plots generated.")
