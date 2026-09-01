import sqlite3
import pandas as pd 

conn = sqlite3.connect('bulletin.db')

raw_folder = 'raw_data'

df = pd.read_csv('raw_data/05_Suppliers.csv')
df.to_sql('05_Suppliers', conn, if_exists='replace', index=False)

conn.close()
print("\nAll tables loaded.")