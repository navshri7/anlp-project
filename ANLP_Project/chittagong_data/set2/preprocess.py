import pandas as pd
df = pd.read_csv('data/set2/dataset.csv')
last_col = df.columns[-1]
df[last_col] = df[last_col].astype(str).apply(lambda x: x.split(',')[0] if ',' in x else x)
df = df[df[last_col] != 'nan']
df.to_json('data/set2/processed_data.json', orient='records', indent=2, force_ascii=False)
print("Data saved to JSON file. Number of records:", len(df))