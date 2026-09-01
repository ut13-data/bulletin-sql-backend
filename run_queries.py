import pandas as pd
import sqlite3

conn = sqlite3.connect("bulletin.db")

query = """
SELECT SUM(LineRevenue * (1 - DiscountPct)) AS total_revenue
FROM FactSalesLines
"""
result = pd.read_sql(query, conn)
print(result)

query2 = """
SELECT
    p.Category,
    SUM(f.LineRevenue * (1 - f.DiscountPct)) AS total_revenue
FROM FactSalesLines f
JOIN DimProduct p ON f.ProductID = p.ProductID
GROUP BY p.Category
ORDER BY total_revenue DESC
"""
result2 = pd.read_sql(query2, conn)
print(result2)

query3 = """
SELECT
    d.FiscalYear,
    SUM(f.LineRevenue * (1 - f.DiscountPct)) AS total_revenue
FROM FactSalesLines f
JOIN DimDate d ON f.OrderDate = d.Date
WHERE d.FiscalYear = 'FY24'
GROUP BY d.FiscalYear
"""

result3 = pd.read_sql(query3, conn)
print(result3)



#inventory turnover
query4 = """
SELECT
    d.Quarter,
    SUM(f.Sold) AS units_sold,
    AVG((f.OpeningStock + f.ClosingStock) / 2.0) AS avg_inventory_units
FROM FactInventoryWeekly f
JOIN DimDate d ON f.WeekEnd = d.Date
GROUP BY d.Quarter
ORDER BY d.Quarter
"""

result4 = pd.read_sql(query4, conn)
result4["turnover_estimate"] = result4["units_sold"] / result4["avg_inventory_units"]
print(result4)

#company-wide Gross Margin % and Gross Profit
query5 = """
SELECT
    SUM(GrossProfit) AS total_gross_profit,
    SUM(GrossProfit) * 100.0 / SUM(Revenue) AS gross_margin_pct
FROM FactFinanceMonthly
"""
result5 = pd.read_sql(query5, conn)
print(result5)

#Gross Profit/Margin % by category
query6 = """
SELECT
    p.Category,
    SUM(f.LineRevenue * (1 - f.DiscountPct)) AS total_revenue,
    SUM(f.Quantity * p.UnitCost) AS total_cogs,
    SUM(f.LineRevenue * (1 - f.DiscountPct)) - SUM(f.Quantity * p.UnitCost) AS gross_profit
FROM FactSalesLines f
JOIN DimProduct p ON f.ProductID = p.ProductID
GROUP BY p.Category
ORDER BY gross_profit DESC
"""
result6 = pd.read_sql(query6, conn)
result6["gross_margin_pct"] = result6["gross_profit"] * 100 / result6["total_revenue"]
print(result6)

#Discount %
query7 = """
SELECT
    SUM(LineRevenue * DiscountPct) * 100.0 / SUM(LineRevenue) AS discount_pct
FROM FactSalesLines
"""
print(pd.read_sql(query7, conn))

#Avg Revenue / Customer
query8 = """
SELECT
    SUM(f.LineRevenue * (1 - f.DiscountPct)) * 1.0 / COUNT(DISTINCT f.CustomerID) AS avg_rev_per_customer
FROM FactSalesLines f
JOIN DimDate d ON f.OrderDate = d.Date
"""
print(pd.read_sql(query8, conn))

#YoY Growth %
query9 = """
SELECT d.FiscalYear, SUM(f.LineRevenue * (1 - f.DiscountPct)) AS revenue
FROM FactSalesLines f
JOIN DimDate d ON f.OrderDate = d.Date
GROUP BY d.FiscalYear
"""
result9 = pd.read_sql(query9, conn)
print(result9)

conn.close()