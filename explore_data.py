import pandas as pd
import os

raw_folder = "raw_data"
files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]

date_keywords = ["date", "weekend", "monthyear"]

for file in files:
    df = pd.read_csv(os.path.join(raw_folder, file))
    likely_date_cols = [col for col in df.columns if any(k in col.lower() for k in date_keywords)]
    if likely_date_cols:
        print(f"{file}: {likely_date_cols}")