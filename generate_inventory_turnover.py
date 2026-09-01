import pandas as pd
import sqlite3
import json

conn = sqlite3.connect("bulletin.db")

# --- Base: per-product average inventory + units sold ---
base = pd.read_sql("""
    SELECT
        f.ProductID,
        p.ProductName,
        p.Category,
        p.ShelfLifeMonths,
        p.UnitCost,
        SUM(f.Sold) AS units_sold,
        AVG((f.OpeningStock + f.ClosingStock) / 2.0) AS avg_inventory_units
    FROM FactInventoryWeekly f
    JOIN DimProduct p ON f.ProductID = p.ProductID
    GROUP BY f.ProductID
""", conn)

base["inventory_turnover"] = base["units_sold"] / base["avg_inventory_units"]
base["warehouse_value"] = base["avg_inventory_units"] * base["UnitCost"]

print(base[["ProductID", "avg_inventory_units", "UnitCost", "warehouse_value"]])

# --- KPIs ---
inventory_turnover_overall = base["units_sold"].sum() / base["avg_inventory_units"].sum()
warehouse_value_total = base["warehouse_value"].sum()

# --- Category Turnover ---
cat = base.groupby("Category").apply(
    lambda g: pd.Series({
        "inventoryTurnover": g["units_sold"].sum() / g["avg_inventory_units"].sum(),
        "avgInventoryValue": g["warehouse_value"].mean(),
    })
).reset_index()

category_turnover = [
    {"category": row["Category"], "inventoryTurnover": round(row["inventoryTurnover"], 2), "avgInventoryValue": round(row["avgInventoryValue"], 2)}
    for _, row in cat.iterrows()
]

# --- Fast Movers (top 5) / Overstock Risks (bottom 5) ---
sorted_base = base.sort_values("inventory_turnover", ascending=False)

fast_movers = [
    {"productId": row["ProductID"], "category": row["Category"], "productName": row["ProductName"], "inventoryTurnover": round(row["inventory_turnover"], 2)}
    for _, row in sorted_base.head(5).iterrows()
]
overstock_risks = [
    {"productId": row["ProductID"], "category": row["Category"], "productName": row["ProductName"], "inventoryTurnover": round(row["inventory_turnover"], 2)}
    for _, row in sorted_base.tail(5).iterrows()
]

# --- Quarterly Turnover (real, company-wide) ---
quarterly = pd.read_sql("""
    SELECT
        d.Quarter,
        SUM(f.Sold) AS units_sold,
        AVG((f.OpeningStock + f.ClosingStock) / 2.0) AS avg_inventory_units
    FROM FactInventoryWeekly f
    JOIN DimDate d ON f.WeekEnd = d.Date
    GROUP BY d.Quarter
    ORDER BY d.Quarter
""", conn)
quarterly["turnover"] = quarterly["units_sold"] / quarterly["avg_inventory_units"]

quarterly_turnover = [
    {"quarter": int(row["Quarter"]), "inventoryTurnover": round(row["turnover"], 2)}
    for _, row in quarterly.iterrows()
]

# --- Shelf Life vs Turnover (bubble chart data) ---
shelf_life_vs_turnover = [
    {
        "productId": row["ProductID"],
        "productName": row["ProductName"],
        "shelfLifeMonths": int(row["ShelfLifeMonths"]),
        "inventoryTurnover": round(row["inventory_turnover"], 2),
        "warehouseValue": round(row["warehouse_value"], 2),
    }
    for _, row in base.iterrows()
]

conn.close()

# --- Write into data.json ---
json_path = "../../balaji-pharma-intelligence/app/inventory-turnover/data.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["inventoryTurnover"] = f"{inventory_turnover_overall:.2f}"
data["averageInventoryValue"] = f"₹{warehouse_value_total / 100_000:.2f}L"
data["categoryTurnover"] = category_turnover
data["fastMovers"] = fast_movers
data["overstockRisks"] = overstock_risks
data["quarterlyTurnover"] = quarterly_turnover
data["shelfLifeVsTurnover"] = shelf_life_vs_turnover

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\nInventory Turnover data.json updated.")
print(f"Inventory Turnover: {data['inventoryTurnover']}")
print(f"Average Inventory Value: {data['averageInventoryValue']}")
print("\nQuarterly Turnover (company-wide):")
print(quarterly_turnover)