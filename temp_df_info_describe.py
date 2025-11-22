import pandas as pd

file_path = 'WA_Fn-UseC_-HR-Employee-Attrition (1).csv'
df = pd.read_csv(file_path)

print("--- df.info() ---")
df.info()

print("\n--- df.describe(include='all') ---")
print(df.describe(include='all').to_string())
