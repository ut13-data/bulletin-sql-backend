import pandas as pd
import sqlite3
import json

conn = sqlite3.connect("bulletin.db")

# --- KPIs: company-wide Gross Margin % and Gross Profit ---
kpis = pd.read_sql("""
    SELECT SUM(GrossProfit) AS total_gross_profit,
           SUM(GrossProfit) * 100.0 / SUM(Revenue) AS gross_margin_pct
    FROM FactFinanceMonthly
""", conn)
total_gross_profit = kpis["total_gross_profit"][0]
gross_margin_pct = kpis["gross_margin_pct"][0]

# --- Category Breakdown (using UnitCost as COGS proxy, same as before) ---
cat = pd.read_sql("""
    SELECT
        p.Category,
        SUM(f.LineRevenue * (1 - f.DiscountPct)) AS total_revenue,
        SUM(f.Quantity * p.UnitCost) AS total_cogs,
        SUM(f.LineRevenue * (1 - f.DiscountPct)) - SUM(f.Quantity * p.UnitCost) AS gross_profit
    FROM FactSalesLines f
    JOIN DimProduct p ON f.ProductID = p.ProductID
    GROUP BY p.Category
    ORDER BY gross_profit DESC
""", conn)
cat["gross_margin_pct"] = cat["gross_profit"] * 100 / cat["total_revenue"]

category_breakdown = [
    {
        "category": row["Category"],
        "grossProfit": round(row["gross_profit"], 2),
        "totalRevenue": round(row["total_revenue"], 2),
        "grossMarginPct": round(row["gross_margin_pct"], 2),
        "totalCogs": round(row["total_cogs"], 2),
    }
    for _, row in cat.iterrows()
]

# --- Quarterly Trend (calendar year, matching original chart) ---
quarterly = pd.read_sql("""
    SELECT
        CAST(strftime('%Y', MonthYear) AS INTEGER) AS year,
        'Qtr ' || ((CAST(strftime('%m', MonthYear) AS INTEGER) - 1) / 3 + 1) AS quarter,
        SUM(GrossProfit) AS gross_profit,
        AVG(GrossMarginPct) * 100 AS gross_margin_pct
    FROM FactFinanceMonthly
    GROUP BY year, quarter
    ORDER BY year, quarter
""", conn)

quarterly_trend = [
    {
        "year": int(row["year"]),
        "quarter": row["quarter"],
        "grossProfit": round(row["gross_profit"], 2),
        "grossMarginPct": round(row["gross_margin_pct"], 2),
        "benchmarkHigh": 35.00,
        "benchmarkLow": 15.00,
    }
    for _, row in quarterly.iterrows()
]

# --- Revenue & COGS by Fiscal Quarter ---
rev_cogs = pd.read_sql("""
    SELECT d.FiscalYear, d.Quarter,
        SUM(f.Revenue) AS total_revenue,
        SUM(f.COGS) AS total_cogs
    FROM FactFinanceMonthly f
    JOIN DimDate d ON f.MonthYear = d.Date
    GROUP BY d.FiscalYear, d.Quarter
    ORDER BY d.FiscalYear, d.Quarter
""", conn)

revenue_cogs_by_fiscal_quarter = [
    {
        "fiscalYear": row["FiscalYear"],
        "quarter": int(row["Quarter"]),
        "totalRevenue": round(row["total_revenue"], 2),
        "totalCogs": round(row["total_cogs"], 2),
    }
    for _, row in rev_cogs.iterrows()
]

conn.close()

# --- Write into data.json ---
json_path = "../../balaji-pharma-intelligence/app/gross-margin/data.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["grossMarginPct"] = f"{gross_margin_pct:.2f}%"
data["grossProfit"] = f"₹{total_gross_profit / 1_000_000:.2f}M"
data["categoryBreakdown"] = category_breakdown
data["quarterlyTrend"] = quarterly_trend
data["revenueCogsByFiscalQuarter"] = revenue_cogs_by_fiscal_quarter

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Gross Margin data.json updated.")
print(f"Gross Margin %: {data['grossMarginPct']}")
print(f"Gross Profit: {data['grossProfit']}")