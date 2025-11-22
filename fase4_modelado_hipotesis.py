import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import os

# --- Configuración ---
DATASET_PATH = 'dataset_hipotesis.csv'
TARGET_VARIABLE = 'Attrition'
PLOTS_DIR = 'plots/hipotesis'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Cargar y Preparar Datos ---
df = pd.read_csv(DATASET_PATH)

X = df.drop(TARGET_VARIABLE, axis=1)
y = df[TARGET_VARIABLE]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Identificar columnas numéricas para escalar
numeric_features = ['Age', 'MonthlyIncome', 'JobSatisfaction']

# Escalar variables numéricas
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(sampling_strategy=0.7, random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Funciones Auxiliares de Visualización ---
def plot_confusion_matrix(y_true, y_pred, model_name, file_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Renuncia', 'Sí Renuncia'],
                yticklabels=['No Renuncia', 'Sí Renuncia'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def plot_decision_tree_graph(model, feature_names, file_path):
    plt.figure(figsize=(20, 12))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=['No Renuncia', 'Sí Renuncia'], rounded=True, fontsize=10)
    plt.title("Árbol de Decisión (Profundidad 3)")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, model_name, file_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Importancia de Variables - {model_name}")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

# --- Modelado y Evaluación ---
models = {
    "Regresión Logística": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE)
}

results = {}
roc_curves = {}

print("--- Iniciando Modelado y Evaluación (Enfoque Hipótesis) ---")

for name, model in models.items():
    # Entrenar
    model.fit(X_train_smote, y_train_smote)
    
    # Predecir
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluar
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {'Accuracy': accuracy, 'Recall': recall, 'AUC': auc}
    
    # Guardar curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    
    print(f"\nResultados para {name}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Recall (Sí Renuncia): {recall:.2f}")
    print(f"  AUC: {auc:.2f}")
    
    # Visualizaciones
    plot_confusion_matrix(y_test, y_pred, name, os.path.join(PLOTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"))
    
    if name == "Árbol de Decisión":
        plot_decision_tree_graph(model, X.columns, os.path.join(PLOTS_DIR, "decision_tree_hipotesis.png"))
        
    if name == "Random Forest":
        plot_feature_importance(model, X.columns, name, os.path.join(PLOTS_DIR, "feature_importance_rf_hipotesis.png"))

# --- Gráfico Comparativo de Curvas ROC ---
plt.figure(figsize=(10, 8))
for name, data in roc_curves.items():
    plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Azar')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Comparación de Curvas ROC (Modelos Hipótesis)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves_comparison_hipotesis.png'), bbox_inches='tight')
plt.close()

print("\n--- Todas las gráficas han sido generadas en la carpeta 'plots/hipotesis/' ---")
