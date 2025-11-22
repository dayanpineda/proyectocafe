import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

# --- 0. Setup ---
# Asegurarse de que el directorio de gráficos exista
if not os.path.exists('plots'):
    os.makedirs('plots')

# --- 1. Carga y Preparación de Datos ---
print("Cargando dataset procesado...")
df = pd.read_csv('dataset_procesado.csv')

# Separar predictores (X) y objetivo (y)
# La variable objetivo es 'Attrition_Yes'
selected_features = [
    'Age',
    'MonthlyIncome',
    'BusinessTravel_Non-Travel',
    'BusinessTravel_Travel_Frequently',
    'BusinessTravel_Travel_Rarely',
    'JobSatisfaction_1',
    'JobSatisfaction_2',
    'JobSatisfaction_3',
    'JobSatisfaction_4',
    'OverTime_No',
    'OverTime_Yes'
]
X = df[selected_features]
y = df['Attrition_Yes']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar las variables numéricas
# Identificar columnas numéricas (no las dummies que ya son 0/1)
numeric_cols = X_train.select_dtypes(include=np.number).columns
# Filtrar solo las que no son dummies (las que tienen valores > 1)
cols_to_scale = [col for col in numeric_cols if X_train[col].max() > 1]

scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Aplicar SMOTE solo al conjunto de entrenamiento
print("Aplicando SMOTE al conjunto de entrenamiento...")
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Función para Graficar Matriz de Confusión ---
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Renuncia', 'Sí Renuncia'], 
                yticklabels=['No Renuncia', 'Sí Renuncia'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

# --- 2. Modelo 1: Regresión Logística ---
print("Entrenando Regresión Logística...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_smote, y_train_smote)
y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

# Métricas y Gráficos
plot_confusion_matrix(y_test, y_pred_log, 'Regresión Logística')
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = roc_auc_score(y_test, y_prob_log)

# --- 3. Modelo 2: Árbol de Decisión ---
print("Entrenando Árbol de Decisión...")
# Usamos max_depth=4 para que el árbol sea visualizable
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_smote, y_train_smote)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# Métricas y Gráficos
plot_confusion_matrix(y_test, y_pred_dt, 'Árbol de Decisión')
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)

# Visualizar el árbol
plt.figure(figsize=(20, 12))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['No Renuncia', 'Sí Renuncia'], rounded=True, fontsize=10)
plt.title("Visualización del Árbol de Decisión (Profundidad Máxima = 4)", fontsize=16)
plt.savefig('plots/decision_tree.png')
plt.close()

# --- 4. Modelo 3: Random Forest ---
print("Entrenando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Métricas y Gráficos
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest')
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

# Gráfico de Importancia de Variables
importances = rf.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15
plt.figure(figsize=(12, 8))
plt.title('Top 15 - Importancia de Variables (Random Forest)')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importancia Relativa')
plt.tight_layout()
plt.savefig('plots/feature_importance_rf.png')
plt.close()

# --- 5. Gráfico Comparativo de Curvas ROC ---
print("Generando gráfico comparativo de Curvas ROC...")
plt.figure(figsize=(10, 8))
plt.plot(fpr_log, tpr_log, label=f'Regresión Logística (AUC = {auc_log:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Árbol de Decisión (AUC = {auc_dt:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Línea de azar
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC Comparativas')
plt.legend()
plt.grid()
plt.savefig('plots/roc_curves_comparison.png')
plt.close()

print("\n--- Resumen de Métricas ---")
print(f"Regresión Logística -> Accuracy: {accuracy_score(y_test, y_pred_log):.2f}, Recall: {recall_score(y_test, y_pred_log):.2f}, AUC: {auc_log:.2f}")
print(f"Árbol de Decisión -> Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}, Recall: {recall_score(y_test, y_pred_dt):.2f}, AUC: {auc_dt:.2f}")
print(f"Random Forest -> Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}, Recall: {recall_score(y_test, y_pred_rf):.2f}, AUC: {auc_rf:.2f}")

print("\n¡Modelado y generación de gráficos completados!")
