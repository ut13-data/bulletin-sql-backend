from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bulletin-balaji-pharma.vercel.app", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_connection():
    return sqlite3.connect("bulletin.db")

@app.get("/")
def read_root():
    return {"message": "bUlleTin backend is running"}

@app.get("/all-data")
def get_all_data():
    conn = get_connection()

    # --- Revenue KPIs ---
    total_rev = pd.read_sql("""
        SELECT SUM(f.LineRevenue * (1 - f.DiscountPct)) AS revenue
        FROM FactSalesLines f
        JOIN DimDate d ON f.OrderDate = d.Date
        WHERE d.FiscalYear = 'FY24'
    """, conn)["revenue"][0]

    discount_pct = pd.read_sql("""
        SELECT SUM(LineRevenue * DiscountPct) * 100.0 / SUM(LineRevenue) AS pct
        FROM FactSalesLines
    """, conn)["pct"][0]

    avg_rev_per_customer = pd.read_sql("""
        SELECT SUM(LineRevenue * (1 - DiscountPct)) * 1.0 / COUNT(DISTINCT CustomerID) AS avg_rev
        FROM FactSalesLines
    """, conn)["avg_rev"][0]

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

    # --- Revenue Chart Data ---
    monthly_trend = pd.read_sql("""
        SELECT
            strftime('%Y-%m', OrderDate) AS month,
            SUM(LineRevenue * (1 - DiscountPct)) AS revenue,
            SUM(LineRevenue * DiscountPct) * 100.0 / SUM(LineRevenue) AS discount_pct
        FROM FactSalesLines
        GROUP BY month
        ORDER BY month
    """, conn)

    monthly_trend["year"] = monthly_trend["month"].str[:4].astype(int)
    monthly_trend["month_num"] = monthly_trend["month"].str[5:7].astype(int)
    monthly_trend_sorted = monthly_trend.sort_values(["year", "month_num"]).reset_index(drop=True)
    monthly_trend_sorted["prior_year_revenue"] = monthly_trend_sorted.groupby("month_num")["revenue"].shift(1)
    monthly_trend_sorted["yoy_growth_pct"] = (
        (monthly_trend_sorted["revenue"] - monthly_trend_sorted["prior_year_revenue"])
        / monthly_trend_sorted["prior_year_revenue"] * 100
    )
    monthly_yoy = monthly_trend_sorted.dropna(subset=["yoy_growth_pct"])

    distributor_share = pd.read_sql("""
        SELECT
            DistributorID AS distributor_id,
            SUM(LineRevenue * (1 - DiscountPct)) AS revenue
        FROM FactSalesLines
        GROUP BY DistributorID
    """, conn)
    distributor_total = distributor_share["revenue"].sum()
    distributor_share["revenue_share_pct"] = distributor_share["revenue"] / distributor_total * 100

    category_share = pd.read_sql("""
        SELECT
            p.Category AS category,
            SUM(f.LineRevenue * (1 - f.DiscountPct)) AS revenue
        FROM FactSalesLines f
        JOIN DimProduct p ON f.ProductID = p.ProductID
        GROUP BY p.Category
    """, conn)
    category_total = category_share["revenue"].sum()
    category_share["revenue_share_pct"] = category_share["revenue"] / category_total * 100
    category_share = category_share.sort_values("revenue_share_pct", ascending=False)

    product_revenue = pd.read_sql("""
        SELECT
            f.ProductID AS product_id,
            p.ProductName AS product_name,
            SUM(f.LineRevenue * (1 - f.DiscountPct)) AS total_revenue
        FROM FactSalesLines f
        JOIN DimProduct p ON f.ProductID = p.ProductID
        GROUP BY f.ProductID, p.ProductName
        ORDER BY total_revenue DESC
    """, conn)

    # --- Gross Margin KPIs ---
    margin_overall = pd.read_sql("""
        SELECT SUM(GrossProfit) AS total_gross_profit,
               SUM(Revenue) AS total_revenue
        FROM FactFinanceMonthly
    """, conn)
    total_gross_profit = margin_overall["total_gross_profit"][0]
    margin_total_revenue = margin_overall["total_revenue"][0]
    gross_margin_pct = (total_gross_profit / margin_total_revenue) * 100

    category_breakdown = pd.read_sql("""
        SELECT
            p.Category AS category,
            SUM(f.LineRevenue * (1 - f.DiscountPct)) AS total_revenue,
            SUM(f.Quantity * p.UnitCost) AS total_cogs
        FROM FactSalesLines f
        JOIN DimProduct p ON f.ProductID = p.ProductID
        GROUP BY p.Category
    """, conn)
    category_breakdown["gross_profit"] = category_breakdown["total_revenue"] - category_breakdown["total_cogs"]
    category_breakdown["gross_margin_pct"] = (
        category_breakdown["gross_profit"] / category_breakdown["total_revenue"] * 100
    )

    quarterly_trend_raw = pd.read_sql("""
        SELECT MonthYear, GrossProfit, Revenue
        FROM FactFinanceMonthly
    """, conn)
    quarterly_trend_raw["year"] = quarterly_trend_raw["MonthYear"].str[:4].astype(int)
    quarterly_trend_raw["month_num"] = quarterly_trend_raw["MonthYear"].str[5:7].astype(int)
    quarterly_trend_raw["quarter"] = ((quarterly_trend_raw["month_num"] - 1) // 3) + 1
    quarterly_grouped = quarterly_trend_raw.groupby(["year", "quarter"]).agg(
        gross_profit=("GrossProfit", "sum"),
        revenue=("Revenue", "sum"),
    ).reset_index()
    quarterly_grouped["gross_margin_pct"] = (quarterly_grouped["gross_profit"] / quarterly_grouped["revenue"]) * 100
        # Revenue vs COGS by fiscal quarter: same COGS calc as the category
    # breakdown (Quantity * UnitCost), joined to DimDate for FiscalYear.
    # Fiscal quarter is derived from the calendar month (fiscal year
    # starts April), NOT taken from DimDate.Quarter directly — that
    # column is calendar-quarter-based and would misalign fiscal Q1
    # with Apr-Jun.
    rev_cogs_raw = pd.read_sql("""
        SELECT
            d.FiscalYear AS fiscal_year,
            d.Month AS month_num,
            f.LineRevenue * (1 - f.DiscountPct) AS revenue,
            f.Quantity * p.UnitCost AS cogs
        FROM FactSalesLines f
        JOIN DimDate d ON f.OrderDate = d.Date
        JOIN DimProduct p ON f.ProductID = p.ProductID
    """, conn)
    rev_cogs_raw["fiscal_quarter"] = ((rev_cogs_raw["month_num"] - 4) % 12) // 3 + 1
    revenue_cogs_by_fiscal_quarter = rev_cogs_raw.groupby(["fiscal_year", "fiscal_quarter"]).agg(
        total_revenue=("revenue", "sum"),
        total_cogs=("cogs", "sum"),
    ).reset_index().sort_values(["fiscal_year", "fiscal_quarter"])
    # --- Inventory Turnover KPIs ---
    inv_raw = pd.read_sql("""
        SELECT
            f.ProductID AS product_id,
            f.ClosingStock AS closing_stock,
            f.Sold AS sold
        FROM FactInventoryWeekly f
    """, conn)
    product_info = pd.read_sql("""
        SELECT ProductID AS product_id, ProductName AS product_name,
               Category AS category, UnitCost AS unit_cost,
               ShelfLifeMonths AS shelf_life_months
        FROM DimProduct
    """, conn)

    inv_per_product = inv_raw.groupby("product_id").agg(
        avg_closing_stock=("closing_stock", "mean"),
        total_sold=("sold", "sum"),
    ).reset_index()
    inv_per_product = inv_per_product.merge(product_info, on="product_id")
    inv_per_product["avg_inventory_value"] = inv_per_product["avg_closing_stock"] * inv_per_product["unit_cost"]
    inv_per_product["cogs"] = inv_per_product["total_sold"] * inv_per_product["unit_cost"]
    inv_per_product["turnover"] = inv_per_product["cogs"] / inv_per_product["avg_inventory_value"]

    total_cogs_inv = inv_per_product["cogs"].sum()
    total_avg_inventory_value = inv_per_product["avg_inventory_value"].sum()
    overall_turnover = total_cogs_inv / total_avg_inventory_value

    category_turnover = inv_per_product.groupby("category").agg(
        cogs=("cogs", "sum"),
        avg_inventory_value=("avg_inventory_value", "sum"),
    ).reset_index()
    category_turnover["turnover"] = category_turnover["cogs"] / category_turnover["avg_inventory_value"]

    fast_movers = inv_per_product.sort_values("turnover", ascending=False).head(5)
    overstock_risks = inv_per_product.sort_values("turnover", ascending=True).head(5)
        # Quarterly turnover: same COGS/Average-Inventory-Value logic as the
    # fleet-level KPI, but split by calendar quarter (blended across all
    # years, matching the existing Qtr 1-4 labels with no year breakdown).
    # IMPORTANT: this recomputes one ratio from SUMMED raw numbers per
    # quarter — it does NOT sum each product's or category's individual
    # turnover ratio together. Summing ratios was the bug in the old
    # static data (~300+ instead of a sane ~50 range), the same class of
    # error as the previously-caught "sum vs mean" and "Warehouse Value"
    # issues — ratios of aggregates, never aggregates of ratios.
    inv_weekly_raw = pd.read_sql("""
        SELECT
            f.ProductID AS product_id,
            f.WeekEnd AS week_end,
            f.ClosingStock AS closing_stock,
            f.Sold AS sold,
            p.UnitCost AS unit_cost
        FROM FactInventoryWeekly f
        JOIN DimProduct p ON f.ProductID = p.ProductID
    """, conn)
    inv_weekly_raw["month_num"] = pd.to_datetime(inv_weekly_raw["week_end"]).dt.month
    inv_weekly_raw["quarter"] = ((inv_weekly_raw["month_num"] - 1) // 3) + 1

    quarterly_per_product = inv_weekly_raw.groupby(["quarter", "product_id"]).agg(
        avg_closing_stock=("closing_stock", "mean"),
        total_sold=("sold", "sum"),
        unit_cost=("unit_cost", "first"),
    ).reset_index()
    quarterly_per_product["avg_inventory_value"] = quarterly_per_product["avg_closing_stock"] * quarterly_per_product["unit_cost"]
    quarterly_per_product["cogs"] = quarterly_per_product["total_sold"] * quarterly_per_product["unit_cost"]

    quarterly_turnover = quarterly_per_product.groupby("quarter").agg(
        cogs=("cogs", "sum"),
        avg_inventory_value=("avg_inventory_value", "sum"),
    ).reset_index()
    quarterly_turnover["turnover"] = quarterly_turnover["cogs"] / quarterly_turnover["avg_inventory_value"]
    conn.close()

    return {
        "revenue": {
            "totalRevenue": f"₹{total_rev / 1_000_000:.2f}M",
            "discountPct": f"{discount_pct:.2f}%",
            "avgRevenuePerCustomer": f"₹{avg_rev_per_customer / 1000:.2f}K",
            "yoyGrowth": f"{yoy_growth_pct:.2f}%",
            "distributorShare": [
                {"distributorId": row["distributor_id"], "revenueSharePct": round(row["revenue_share_pct"], 2)}
                for _, row in distributor_share.iterrows()
            ],
            "categoryShare": [
                {"category": row["category"], "revenueSharePct": round(row["revenue_share_pct"], 2)}
                for _, row in category_share.iterrows()
            ],
            "monthlyRevenueTrend": [
                {"month": row["month"], "revenue": round(row["revenue"], 2), "discountPct": round(row["discount_pct"], 2)}
                for _, row in monthly_trend_sorted.iterrows()
            ],
            "monthlyYoyGrowth": [
                {"month": row["month"], "yoyGrowthPct": round(row["yoy_growth_pct"], 2)}
                for _, row in monthly_yoy.iterrows()
            ],
            "productRevenue": [
                {"productId": row["product_id"], "productName": row["product_name"], "totalRevenue": round(row["total_revenue"], 2)}
                for _, row in product_revenue.iterrows()
            ],
        },
        "grossMargin": {
            "grossMarginPct": f"{gross_margin_pct:.2f}%",
            "grossProfit": f"₹{total_gross_profit / 1_000_000:.2f}M",
            "categoryBreakdown": [
                {
                    "category": row["category"],
                    "grossProfit": round(row["gross_profit"], 2),
                    "totalRevenue": round(row["total_revenue"], 2),
                    "grossMarginPct": round(row["gross_margin_pct"], 2),
                    "totalCogs": round(row["total_cogs"], 2),
                }
                for _, row in category_breakdown.iterrows()
            ],
            "quarterlyTrend": [
                {
                    "year": int(row["year"]),
                    "quarter": f"Qtr {int(row['quarter'])}",
                    "grossProfit": round(row["gross_profit"], 2),
                    "grossMarginPct": round(row["gross_margin_pct"], 2),
                    "benchmarkHigh": 35.0,
                    "benchmarkLow": 15.0,
                }
                for _, row in quarterly_grouped.iterrows()
            ],
                "revenueCogsByFiscalQuarter": [
                {
                    "fiscalYear": row["fiscal_year"],
                    "quarter": int(row["fiscal_quarter"]),
                    "totalRevenue": round(row["total_revenue"], 2),
                    "totalCogs": round(row["total_cogs"], 2),
                }
                for _, row in revenue_cogs_by_fiscal_quarter.iterrows()
            ],
        },
        "inventoryTurnover": {
            "inventoryTurnover": f"{overall_turnover:.2f}",
            "averageInventoryValue": f"₹{total_avg_inventory_value / 100000:.2f}L",
            "productTurnover": [
                {"productId": row["product_id"], "inventoryTurnover": round(row["turnover"], 2)}
                for _, row in inv_per_product.iterrows()
            ],
            "categoryTurnover": [
                {"category": row["category"], "inventoryTurnover": round(row["turnover"], 2), "avgInventoryValue": round(row["avg_inventory_value"], 2)}
                for _, row in category_turnover.iterrows()
            ],
            "fastMovers": [
                {"productId": row["product_id"], "category": row["category"], "productName": row["product_name"], "inventoryTurnover": round(row["turnover"], 2)}
                for _, row in fast_movers.iterrows()
            ],
            "overstockRisks": [
                {"productId": row["product_id"], "category": row["category"], "productName": row["product_name"], "inventoryTurnover": round(row["turnover"], 2)}
                for _, row in overstock_risks.iterrows()
            ],
            "shelfLifeVsTurnover": [
                {
                    "productId": row["product_id"],
                    "productName": row["product_name"],
                    "shelfLifeMonths": int(row["shelf_life_months"]),
                    "inventoryTurnover": round(row["turnover"], 2),
                    "warehouseValue": round(row["avg_inventory_value"], 2),
                               
                }
                for _, row in inv_per_product.iterrows()
            ],
             "quarterlyTurnover": [
                     {
                        "quarter": int(row["quarter"]),
                        "inventoryTurnover": round(row["turnover"], 2),
                      }
                     for _, row in quarterly_turnover.iterrows()
             ],
        }
    }