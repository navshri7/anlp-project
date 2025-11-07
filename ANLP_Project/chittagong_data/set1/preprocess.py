import pandas as pd
df = pd.read_excel('data/set1/Original_to_Dialect.xlsx')
cols_to_remove = ['Total Number of Translation', 'Percentage of Translation']
df = df.drop(columns=cols_to_remove)
last_col = df.columns[-1]
df[last_col] = df[last_col].astype(str).apply(lambda x: x.split(',')[0] if ',' in x else x)
df = df[df[last_col] != 'nan']
df.to_json('data/set1/processed_data.json', orient='records', indent=2, force_ascii=False)
print("Data saved to JSON file. Number of records:", len(df))