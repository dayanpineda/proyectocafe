import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.ticker as mtick

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

# 1. Load and Prepare Data (Same as before)
print("Loading data...")
df = pd.read_csv('arabica_data_cleaned.csv')
df['HighQuality'] = (df['Total.Cup.Points'] >= 84).astype(int)

features = [
    'Country.of.Origin', 'Region', 'Processing.Method', 'Moisture', 
    'Category.One.Defects', 'Color', 'Category.Two.Defects', 'altitude_mean_meters'
]

X = df[features]
y = df['HighQuality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Train Best Model (Random Forest for robust probability estimation)
numeric_features = ['Moisture', 'Category.One.Defects', 'Category.Two.Defects', 'altitude_mean_meters']
categorical_features = ['Country.of.Origin', 'Region', 'Processing.Method', 'Color']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

clf = ImbPipeline(steps=[('preprocessor', preprocessor),
                         ('smote', SMOTE(random_state=42)),
                         ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

print("Training model...")
clf.fit(X_train, y_train)

# 3. Generate Evaluation Data
y_prob = clf.predict_proba(X_test)[:, 1]
df_eval = pd.DataFrame({'Actual': y_test, 'Probability': y_prob})
df_eval = df_eval.sort_values('Probability', ascending=False).reset_index(drop=True)

# 4. Cumulative Gains Chart (Lift Chart)
# Calculate cumulative sum of actual positives
df_eval['CumulativePositives'] = df_eval['Actual'].cumsum()
# Calculate total positives
total_positives = df_eval['Actual'].sum()
# Calculate percentage of positives captured
df_eval['PercentPositivesCaptured'] = df_eval['CumulativePositives'] / total_positives
# Calculate percentage of sample contacted
df_eval['PercentSampleContacted'] = (df_eval.index + 1) / len(df_eval)

plt.figure(figsize=(10, 6))
plt.plot(df_eval['PercentSampleContacted'], df_eval['PercentPositivesCaptured'], label='Modelo (Random Forest)', color='#4a00e0', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Selección Aleatoria', color='gray')
plt.title('Gráfico de Ganancia Acumulada (Cumulative Gains)', color='white')
plt.xlabel('% de Muestras Analizadas (Ordenadas por Probabilidad)', color='white')
plt.ylabel('% de Café de Alta Calidad Encontrado', color='white')
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend()
plt.savefig('plots_cafe/cumulative_gains.png')
plt.close()

# 5. Cost-Benefit Analysis Plot
# Scenario:
# Cost of Cupping = $50 per sample
# Profit from High Quality Lot = $500
# Strategy A: Cup Everything (Baseline)
# Strategy B: Cup only Top X% predicted by model

cost_per_cupping = 50
profit_per_high_quality = 500

# Calculate ROI for different thresholds
thresholds = np.linspace(0, 1, 100)
rois = []
profits = []

total_samples = len(df_eval)

for p in df_eval['PercentSampleContacted']:
    # Number of samples to cup at this percentage
    n_samples = int(p * total_samples)
    if n_samples == 0:
        rois.append(0)
        profits.append(0)
        continue
        
    # Positives found in this top slice
    n_positives = df_eval.iloc[:n_samples]['Actual'].sum()
    
    cost = n_samples * cost_per_cupping
    revenue = n_positives * profit_per_high_quality
    profit = revenue - cost
    roi = (profit / cost) * 100 if cost > 0 else 0
    
    profits.append(profit)

# Plot Profit Curve
plt.figure(figsize=(10, 6))
plt.plot(df_eval['PercentSampleContacted'], profits, color='#00ff88', linewidth=2)
plt.title('Análisis de Rentabilidad: Estrategia de Selección', color='white')
plt.xlabel('% de Muestras Seleccionadas para Cata', color='white')
plt.ylabel('Beneficio Estimado ($)', color='white')
plt.axvline(x=df_eval['PercentSampleContacted'][np.argmax(profits)], color='white', linestyle='--', label=f'Punto Óptimo (~{int(df_eval["PercentSampleContacted"][np.argmax(profits)]*100)}%)')
plt.legend()
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.savefig('plots_cafe/profit_analysis.png')
plt.close()

print("Phase 5 plots generated successfully.")
