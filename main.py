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
    # Monthly trend: revenue + discount% per calendar month, across all years
    # (not just FY24), since the Explore tab shows full history.
    monthly_trend = pd.read_sql("""
        SELECT
            strftime('%Y-%m', OrderDate) AS month,
            SUM(LineRevenue * (1 - DiscountPct)) AS revenue,
            SUM(LineRevenue * DiscountPct) * 100.0 / SUM(LineRevenue) AS discount_pct
        FROM FactSalesLines
        GROUP BY month
        ORDER BY month
    """, conn)

    # YoY monthly growth: compare each month's revenue to the same month
    # one year earlier. Done in pandas (not SQL) since it's a row-to-row
    # comparison across a shifted index, which pandas handles more simply
    # than a SQL self-join here.
    monthly_trend["year"] = monthly_trend["month"].str[:4].astype(int)
    monthly_trend["month_num"] = monthly_trend["month"].str[5:7].astype(int)
    monthly_trend_sorted = monthly_trend.sort_values(["year", "month_num"]).reset_index(drop=True)
    monthly_trend_sorted["prior_year_revenue"] = monthly_trend_sorted.groupby("month_num")["revenue"].shift(1)
    monthly_trend_sorted["yoy_growth_pct"] = (
        (monthly_trend_sorted["revenue"] - monthly_trend_sorted["prior_year_revenue"])
        / monthly_trend_sorted["prior_year_revenue"] * 100
    )
    # Only months with a valid prior-year comparison are shown (matches
    # the Flagged YoY measure blanking out on partial/missing prior periods).
    monthly_yoy = monthly_trend_sorted.dropna(subset=["yoy_growth_pct"])

    # Distributor share: revenue split by distributor as % of total.
    distributor_share = pd.read_sql("""
        SELECT
            DistributorID AS distributor_id,
            SUM(LineRevenue * (1 - DiscountPct)) AS revenue
        FROM FactSalesLines
        GROUP BY DistributorID
    """, conn)
    distributor_total = distributor_share["revenue"].sum()
    distributor_share["revenue_share_pct"] = distributor_share["revenue"] / distributor_total * 100

    # Category share: revenue split by product category as % of total.
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

    # Product revenue: total revenue per SKU, highest first.
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
    # Overall margin: FactFinanceMonthly already stores GrossProfit and
    # Revenue at month grain, so we sum both and recompute the % rather
    # than averaging the monthly percentages (averaging % would weight
    # every month equally regardless of size, which distorts the true
    # blended margin).
    margin_overall = pd.read_sql("""
        SELECT SUM(GrossProfit) AS total_gross_profit,
               SUM(Revenue) AS total_revenue
        FROM FactFinanceMonthly
    """, conn)
    total_gross_profit = margin_overall["total_gross_profit"][0]
    margin_total_revenue = margin_overall["total_revenue"][0]
    gross_margin_pct = (total_gross_profit / margin_total_revenue) * 100

    # Category breakdown: FactFinanceMonthly has no category column, so
    # this comes from FactSalesLines (revenue) joined to DimProduct
    # (Category + UnitCost, to derive COGS at line level).
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

    # Quarterly trend: FactFinanceMonthly.MonthYear is calendar-month
    # (assumed format 'YYYY-MM'), so calendar quarter is derived directly
    # from the month number rather than needing a DimDate join.
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

    conn.close()

    return {
        "revenue": {
            "totalRevenue": f"₹{total_rev / 1_000_000:.2f}M",
            "discountPct": f"{discount_pct:.2f}%",
            "avgRevenuePerCustomer": f"₹{avg_rev_per_customer / 1000:.2f}K",
            "yoyGrowth": f"{yoy_growth_pct:.2f}%",
            "distributorShare": [
                {
                    "distributorId": row["distributor_id"],
                    "revenueSharePct": round(row["revenue_share_pct"], 2),
                }
                for _, row in distributor_share.iterrows()
            ],
            "categoryShare": [
                {
                    "category": row["category"],
                    "revenueSharePct": round(row["revenue_share_pct"], 2),
                }
                for _, row in category_share.iterrows()
            ],
            "monthlyRevenueTrend": [
                {
                    "month": row["month"],
                    "revenue": round(row["revenue"], 2),
                    "discountPct": round(row["discount_pct"], 2),
                }
                for _, row in monthly_trend_sorted.iterrows()
            ],
            "monthlyYoyGrowth": [
                {
                    "month": row["month"],
                    "yoyGrowthPct": round(row["yoy_growth_pct"], 2),
                }
                for _, row in monthly_yoy.iterrows()
            ],
            "productRevenue": [
                {
                    "productId": row["product_id"],
                    "productName": row["product_name"],
                    "totalRevenue": round(row["total_revenue"], 2),
                }
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
            # revenueCogsByFiscalQuarter intentionally left out for now —
            # needs your DimDate fiscal-quarter join confirmed first.
        },
        "inventoryTurnover": {}
    }