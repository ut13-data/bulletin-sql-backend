import pandas as pd
import sqlite3
import os

conn = sqlite3.connect("bulletin.db")
raw_folder = "raw_data"

# Maps filename -> list of columns that should be parsed as dates
date_columns = {
    "DimEmployee.csv": ["JoinDate"],
    "DimProduct.csv": ["LaunchDate"],
    "DimPromotion.csv": ["StartDate", "EndDate"],
    "DimDate.csv": ["Date"],
    "DimCustomer.csv": ["OnboardDate", "ClosureDate"],
    "DimDistributor.csv": ["OnboardDate"],
    "FactFinanceMonthly.csv": ["MonthYear"],
    "FactInventoryWeekly.csv": ["WeekEnd"],
    "FactProductionBatches.csv": ["ProductionWeekEnd"],
    "FactPurchaseOrderLines.csv": ["OrderDate", "ExpectedDeliveryDate", "ActualDeliveryDate"],
    "FactRawMaterialInventoryWeekly.csv": ["WeekEnd"],
    "FactSalesLines.csv": ["OrderDate"],
    "FactSalesOrders.csv": ["OrderDate"],
}

files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]

for file in files:
    table_name = file.replace(".csv", "")
    parse_cols = date_columns.get(file, [])

    df = pd.read_csv(os.path.join(raw_folder, file), parse_dates=parse_cols)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"Loaded {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")

conn.close()
print("\nAll tables loaded.")