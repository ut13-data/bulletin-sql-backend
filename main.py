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

    conn.close()

    return {
        "revenue": {
            "totalRevenue": f"₹{total_rev / 1_000_000:.2f}M",
            "discountPct": f"{discount_pct:.2f}%",
            "avgRevenuePerCustomer": f"₹{avg_rev_per_customer / 1000:.2f}K",
            "yoyGrowth": f"{yoy_growth_pct:.2f}%"
        },
        "grossMargin": {},
        "inventoryTurnover": {}
    }

    return {
        "revenue": {
            "totalRevenue": f"₹{total_rev / 1_000_000:.2f}M",
            "discountPct": f"{discount_pct:.2f}%",
            "avgRevenuePerCustomer": f"₹{avg_rev_per_customer / 1000:.2f}K",
            "yoyGrowth": f"{yoy_growth_pct:.2f}%"
        },
        "grossMargin": {},
        "inventoryTurnover": {}
    }