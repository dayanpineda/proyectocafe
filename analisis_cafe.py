import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create plots directory
if not os.path.exists('plots_cafe'):
    os.makedirs('plots_cafe')

# 1. Load Data
print("Loading data...")
df = pd.read_csv('arabica_data_cleaned.csv')

# 2. Define Target Variable (High Quality >= 84)
df['HighQuality'] = (df['Total.Cup.Points'] >= 84).astype(int)
print(f"Class distribution:\n{df['HighQuality'].value_counts(normalize=True)}")

# 3. Feature Selection (Avoid Leakage)
# Exclude sensory scores that directly contribute to Total.Cup.Points
drop_cols = [
    'Unnamed: 0', 'Species', 'Owner', 'Farm.Name', 'Lot.Number', 'Mill', 'ICO.Number', 
    'Company', 'Producer', 'Number.of.Bags', 'Bag.Weight', 'In.Country.Partner', 
    'Harvest.Year', 'Grading.Date', 'Owner.1', 'Variety', 'Expiration', 
    'Certification.Body', 'Certification.Address', 'Certification.Contact', 
    'unit_of_measurement', 'altitude_low_meters', 'altitude_high_meters',
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 
    'Clean.Cup', 'Sweetness', 'Cupper.Points', 'Total.Cup.Points', 
    'Quakers' # Too many missing values often
]

# Keep relevant features
features = [
    'Country.of.Origin', 'Region', 'Processing.Method', 'Moisture', 
    'Category.One.Defects', 'Color', 'Category.Two.Defects', 'altitude_mean_meters'
]

df_model = df[features + ['HighQuality']].copy()

# 4. EDA Plots
print("Generating EDA plots...")
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['text.color'] = '#f0f2f5'
plt.rcParams['axes.labelcolor'] = '#f0f2f5'
plt.rcParams['xtick.color'] = '#f0f2f5'
plt.rcParams['ytick.color'] = '#f0f2f5'

# Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='HighQuality', data=df_model, palette=['#4a00e0', '#8e2de2'])
plt.title('Distribución de Calidad (0=Normal, 1=Alta)', color='white')
plt.savefig('plots_cafe/quality_distribution.png')
plt.close()

# Altitude Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df_model, x='altitude_mean_meters', hue='HighQuality', kde=True, palette=['#4a00e0', '#8e2de2'])
plt.title('Distribución de Altitud por Calidad', color='white')
plt.xlim(0, 3000) # Limit outliers
plt.savefig('plots_cafe/altitude_distribution.png')
plt.close()

# Moisture vs Quality
plt.figure(figsize=(10, 6))
sns.boxplot(x='HighQuality', y='Moisture', data=df_model, palette=['#4a00e0', '#8e2de2'])
plt.title('Humedad vs Calidad', color='white')
plt.savefig('plots_cafe/moisture_vs_quality.png')
plt.close()

# Defects vs Quality
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Category.Two.Defects', y='Category.One.Defects', hue='HighQuality', data=df_model, palette=['#4a00e0', '#8e2de2'], alpha=0.6)
plt.title('Defectos vs Calidad', color='white')
plt.savefig('plots_cafe/defects_vs_quality.png')
plt.close()

# 5. Preprocessing
print("Preprocessing...")
X = df_model.drop('HighQuality', axis=1)
y = df_model['HighQuality']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Transformers
numeric_features = ['Moisture', 'Category.One.Defects', 'Category.Two.Defects', 'altitude_mean_meters']
categorical_features = ['Country.of.Origin', 'Region', 'Processing.Method', 'Color']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Modeling
print("Training models...")

models = {
    'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Árbol de Decisión': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Pipeline with SMOTE
    clf = ImbPipeline(steps=[('preprocessor', preprocessor),
                             ('smote', SMOTE(random_state=42)),
                             ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {'Accuracy': acc, 'Recall': rec, 'AUC': auc, 'Model': clf}
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
    plt.title(f'Matriz de Confusión - {name}', color='white')
    plt.ylabel('Real', color='white')
    plt.xlabel('Predicho', color='white')
    plt.savefig(f'plots_cafe/confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()

# 7. Model Evaluation Plots
print("Generating evaluation plots...")

# ROC Curves
plt.figure(figsize=(10, 8))
for name, res in results.items():
    model = res['Model']
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {res["AUC"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos', color='white')
plt.ylabel('Tasa de Verdaderos Positivos', color='white')
plt.title('Curvas ROC Comparativas', color='white')
plt.legend()
plt.savefig('plots_cafe/roc_curves_comparison.png')
plt.close()

# Feature Importance (Random Forest)
rf_model = results['Random Forest']['Model'].named_steps['classifier']
feature_names = (results['Random Forest']['Model'].named_steps['preprocessor']
                 .transformers_[1][1]
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_features))
all_features = numeric_features + list(feature_names)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10] # Top 10

plt.figure(figsize=(10, 6))
plt.title('Importancia de Variables - Random Forest (Top 10)', color='white')
plt.bar(range(10), importances[indices], align='center', color='#8e2de2')
plt.xticks(range(10), [all_features[i] for i in indices], rotation=45, ha='right', color='white')
plt.tight_layout()
plt.savefig('plots_cafe/feature_importance_rf.png')
plt.close()

# Decision Tree Visualization
dt_model = results['Árbol de Decisión']['Model'].named_steps['classifier']
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=all_features, filled=True, rounded=True, max_depth=3, fontsize=10)
plt.title('Árbol de Decisión (Simplificado)', color='black') # Tree text is usually black
plt.savefig('plots_cafe/decision_tree.png')
plt.close()

# Save processed dataset
X_processed = preprocessor.fit_transform(X)
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()

# Get feature names from the preprocessor fitted on the full dataset
new_feature_names = (preprocessor.named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_features))
final_features = numeric_features + list(new_feature_names)

df_processed = pd.DataFrame(X_processed, columns=final_features)
df_processed['HighQuality'] = y.reset_index(drop=True)
df_processed.to_csv('dataset_cafe_procesado.csv', index=False)

print("Analysis complete!")
print(results)
