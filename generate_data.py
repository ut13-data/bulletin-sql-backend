import pandas as pd
import sqlite3
import json

conn = sqlite3.connect("bulletin.db")

# --- Total Revenue (FY24) ---
total_rev = pd.read_sql("""
    SELECT SUM(f.LineRevenue * (1 - f.DiscountPct)) AS revenue
    FROM FactSalesLines f
    JOIN DimDate d ON f.OrderDate = d.Date
    WHERE d.FiscalYear = 'FY24'
""", conn)["revenue"][0]

# --- Discount % (all data) ---
discount_pct = pd.read_sql("""
    SELECT SUM(LineRevenue * DiscountPct) * 100.0 / SUM(LineRevenue) AS pct
    FROM FactSalesLines
""", conn)["pct"][0]

# --- Avg Revenue per Customer (all data) ---
avg_rev_per_customer = pd.read_sql("""
    SELECT SUM(LineRevenue * (1 - DiscountPct)) * 1.0 / COUNT(DISTINCT CustomerID) AS avg_rev
    FROM FactSalesLines
""", conn)["avg_rev"][0]

# --- YoY Growth (FY24 vs FY23) ---
yoy = pd.read_sql("""
    SELECT d.FiscalYear, SUM(f.LineRevenue * (1 - f.DiscountPct)) AS revenue
    FROM FactSalesLines f
    JOIN DimDate d ON f.OrderDate = d.Date
    WHERE d.FiscalYear IN ('FY23', 'FY24')
    GROUP BY d.FiscalYear
""", conn)

fy23_rev = yoy[yoy["FiscalYear"] == "FY23"]["revenue"].values[0]
fy24_rev = yoy[yoy["FiscalYear"] == "FY24"]["revenue"].values[0]
yoy_growth_pct = (fy24_rev - fy23_rev) / fy23_rev * 100

# --- Monthly Revenue + Discount Trend (all months) ---
monthly = pd.read_sql("""
    SELECT
        strftime('%Y-%m', OrderDate) AS month,
        SUM(LineRevenue * (1 - DiscountPct)) AS revenue,
        SUM(LineRevenue * DiscountPct) * 100.0 / SUM(LineRevenue) AS discount_pct
    FROM FactSalesLines
    GROUP BY month
    ORDER BY month
""", conn)

monthly_revenue_trend = [
    {"month": row["month"], "revenue": round(row["revenue"], 2), "discountPct": round(row["discount_pct"], 2)}
    for _, row in monthly.iterrows()
]

# --- Monthly YoY Growth ---
monthly["year"] = monthly["month"].str[:4].astype(int)
monthly["mo"] = monthly["month"].str[5:7]

monthly_yoy = []
for _, row in monthly.iterrows():
    prior_year_month = f"{row['year'] - 1}-{row['mo']}"
    prior = monthly[monthly["month"] == prior_year_month]
    if not prior.empty:
        prior_rev = prior["revenue"].values[0]
        growth = (row["revenue"] - prior_rev) / prior_rev * 100
        monthly_yoy.append({"month": row["month"], "yoyGrowthPct": round(growth, 2)})

# --- Distributor Share ---
dist = pd.read_sql("""
    SELECT DistributorID,
        SUM(LineRevenue * (1 - DiscountPct)) * 100.0 /
        (SELECT SUM(LineRevenue * (1 - DiscountPct)) FROM FactSalesLines) AS share_pct
    FROM FactSalesLines
    GROUP BY DistributorID
""", conn)
distributor_share = [
    {"distributorId": row["DistributorID"], "revenueSharePct": round(row["share_pct"], 2)}
    for _, row in dist.iterrows()
]

# --- Category Share ---
cat = pd.read_sql("""
    SELECT p.Category,
        SUM(f.LineRevenue * (1 - f.DiscountPct)) * 100.0 /
        (SELECT SUM(LineRevenue * (1 - DiscountPct)) FROM FactSalesLines) AS share_pct
    FROM FactSalesLines f
    JOIN DimProduct p ON f.ProductID = p.ProductID
    GROUP BY p.Category
    ORDER BY share_pct DESC
""", conn)
category_share = [
    {"category": row["Category"], "revenueSharePct": round(row["share_pct"], 2)}
    for _, row in cat.iterrows()
]

conn.close()

# --- Load existing data.json ---
json_path = "../../balaji-pharma-intelligence/app/revenue/data.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Update all fields ---
data["totalRevenue"] = f"₹{total_rev / 1_000_000:.2f}M"
data["discountPct"] = f"{discount_pct:.2f}%"
data["avgRevenuePerCustomer"] = f"₹{avg_rev_per_customer / 1000:.2f}K"
data["yoyGrowth"] = f"{yoy_growth_pct:.2f}%"
data["monthlyRevenueTrend"] = monthly_revenue_trend
data["monthlyYoyGrowth"] = monthly_yoy
data["distributorShare"] = distributor_share
data["categoryShare"] = category_share

# --- Write it back ---
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("data.json updated successfully.")
print(f"Total Revenue: {data['totalRevenue']}")
print(f"Discount %: {data['discountPct']}")
print(f"Avg Revenue/Customer: {data['avgRevenuePerCustomer']}")
print(f"YoY Growth: {data['yoyGrowth']}")