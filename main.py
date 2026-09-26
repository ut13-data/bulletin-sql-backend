"""
bUlleTin FastAPI backend.

All dashboard numbers now come from agent/metrics.py, the same metric layer
the Ask bUlleTin agent uses, so a KPI card and a chat answer can never disagree.

JSON keys are unchanged, so existing frontend pages keep working. What changed
in the NUMBERS (see CHANGES.md for details):
  * Gross margin uses net revenue (after discounts) and actual production cost.
    Category and quarterly margins now reconcile with the headline margin.
  * Inventory turnover is annualised, so it agrees with DIO (turnover x DIO = 365).
  * Inventory is valued at production cost, the same basis as COGS.
  * Quarterly turnover has one point per year-quarter (new "year" and "label"
    fields), no longer mixing the same quarter from different years.
  * Revenue KPI cards all use the latest complete fiscal year.
"""
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from agent import metrics as M                      # noqa: E402  (env must be loaded first)
from agent.db import read_sql                        # noqa: E402
from agent.config import MAX_TURNS                  # noqa: E402
from agent.periods import resolve                   # noqa: E402

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bulletin-balaji-pharma.vercel.app", "http://localhost:3000"],
    allow_origin_regex=r"https://bulletin-balaji-pharma.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Results are cached per process: bulletin.db only changes on redeploy.
_cache: dict = {}


def cached(key):
    def wrap(fn):
        def inner():
            if key not in _cache:
                _cache[key] = fn()
            return _cache[key]
        return inner
    return wrap


def r2(x) -> float:
    return round(float(x), 2)


def lakh(x) -> str:
    return f"₹{x / 1e5:.2f}L"


def million(x) -> str:
    return f"₹{x / 1e6:.2f}M"


@app.get("/")
def read_root():
    return {"message": "bUlleTin backend is running"}


# ============================================================
# REVENUE
# ============================================================

@cached("revenue")
def get_revenue_data():
    fy = resolve({"type": "fiscal_year", "value": "latest_complete"})             # FY24 with current data
    fy_prev = resolve({"type": "fiscal_year", "value": f"FY{(fy.start.year) % 100:02d}"})
    all_data = resolve({"type": "all"})

    cards = M.compute(["net_revenue", "discount_pct", "revenue_per_customer"], fy).df.iloc[0]
    prev_rev = M.value("net_revenue", fy_prev)
    yoy_growth_pct = (cards.net_revenue - prev_rev) / prev_rev * 100

    monthly = M.compute(["net_revenue", "discount_pct"], all_data, grain="month").df
    by_month = monthly.set_index("period")["net_revenue"]
    monthly_yoy = []
    for month, rev in by_month.items():
        prior = f"{int(month[:4]) - 1}{month[4:]}"
        if prior in by_month.index:
            monthly_yoy.append({"month": month, "yoyGrowthPct": r2((rev - by_month[prior]) / by_month[prior] * 100)})

    dist = M.compute(["net_revenue"], all_data, dims=["distributor"]).df
    dist_ids = read_sql("SELECT DistributorID, DistributorName FROM DimDistributor").set_index("DistributorName")["DistributorID"]
    cat = M.compute(["net_revenue"], all_data, dims=["category"]).df.sort_values("net_revenue", ascending=False)
    prod = M.compute(["net_revenue"], all_data, dims=["product"]).df.sort_values("net_revenue", ascending=False)
    prod_ids = read_sql("SELECT ProductID, ProductName FROM DimProduct").set_index("ProductName")["ProductID"]

    return {
        "period": fy.short,
        "totalRevenue": million(cards.net_revenue),
        "discountPct": f"{cards.discount_pct:.2f}%",
        "avgRevenuePerCustomer": f"₹{cards.revenue_per_customer / 1000:.2f}K",
        "yoyGrowth": f"{yoy_growth_pct:.2f}%",
        "distributorShare": [{"distributorId": dist_ids.get(r.distributor, r.distributor),
                              "revenueSharePct": r2(r.net_revenue / dist.net_revenue.sum() * 100)}
                             for r in dist.itertuples()],
        "categoryShare": [{"category": r.category, "revenueSharePct": r2(r.net_revenue / cat.net_revenue.sum() * 100)}
                          for r in cat.itertuples()],
        "monthlyRevenueTrend": [{"month": r.period, "revenue": r2(r.net_revenue), "discountPct": r2(r.discount_pct)}
                                for r in monthly.itertuples()],
        "monthlyYoyGrowth": monthly_yoy,
        "productRevenue": [{"productId": prod_ids.get(r.product, r.product), "productName": r.product,
                            "totalRevenue": r2(r.net_revenue)} for r in prod.itertuples()],
    }


# ============================================================
# GROSS MARGIN
# ============================================================

@cached("grossMargin")
def get_gross_margin_data():
    all_data = resolve({"type": "all"})
    keys = ["net_revenue", "cogs", "gross_profit", "gross_margin_pct"]
    total = M.compute(keys, all_data).df.iloc[0]
    cat = M.compute(keys, all_data, dims=["category"]).df
    qtr = M.compute(keys, all_data, grain="quarter").df
    fq = M.compute(keys, all_data, grain="fiscal_quarter").df

    return {
        "grossMarginPct": f"{total.gross_margin_pct:.2f}%",
        "grossProfit": million(total.gross_profit),
        "categoryBreakdown": [{"category": r.category, "grossProfit": r2(r.gross_profit), "totalRevenue": r2(r.net_revenue),
                               "grossMarginPct": r2(r.gross_margin_pct), "totalCogs": r2(r.cogs)} for r in cat.itertuples()],
        "quarterlyTrend": [{"year": int(r.period[:4]), "quarter": f"Qtr {r.period[-1]}", "grossProfit": r2(r.gross_profit),
                            "grossMarginPct": r2(r.gross_margin_pct), "benchmarkHigh": 35.0, "benchmarkLow": 15.0}
                           for r in qtr.itertuples()],
        "revenueCogsByFiscalQuarter": [{"fiscalYear": r.period[:4], "quarter": int(r.period[-1]),
                                        "totalRevenue": r2(r.net_revenue), "totalCogs": r2(r.cogs)} for r in fq.itertuples()],
    }


# ============================================================
# INVENTORY TURNOVER
# ============================================================

@cached("inventoryTurnover")
def get_inventory_turnover_data():
    all_data = resolve({"type": "all"})
    keys = ["inventory_turnover", "avg_inventory_value"]
    total = M.compute(keys, all_data).df.iloc[0]
    prod = M.compute(keys, all_data, dims=["product"]).df
    info = read_sql("SELECT ProductID, ProductName, Category, ShelfLifeMonths FROM DimProduct")
    prod = prod.merge(info, left_on="product", right_on="ProductName")
    cat = M.compute(keys, all_data, dims=["category"]).df
    qtr = M.compute(["inventory_turnover"], all_data, grain="quarter").df

    def mover(r):
        return {"productId": r.ProductID, "category": r.Category, "productName": r.ProductName,
                "inventoryTurnover": r2(r.inventory_turnover)}

    ranked = prod.sort_values("inventory_turnover", ascending=False)
    return {
        "inventoryTurnover": f"{total.inventory_turnover:.2f}",
        "averageInventoryValue": lakh(total.avg_inventory_value),
        "productTurnover": [{"productId": r.ProductID, "inventoryTurnover": r2(r.inventory_turnover)} for r in prod.itertuples()],
        "categoryTurnover": [{"category": r.category, "inventoryTurnover": r2(r.inventory_turnover),
                              "avgInventoryValue": r2(r.avg_inventory_value)} for r in cat.itertuples()],
        "fastMovers": [mover(r) for r in ranked.head(5).itertuples()],
        "overstockRisks": [mover(r) for r in ranked.tail(5).iloc[::-1].itertuples()],
        "shelfLifeVsTurnover": [{"productId": r.ProductID, "productName": r.ProductName,
                                 "shelfLifeMonths": int(r.ShelfLifeMonths), "inventoryTurnover": r2(r.inventory_turnover),
                                 "warehouseValue": r2(r.avg_inventory_value)} for r in prod.itertuples()],
        # One point per year-quarter now (the old version merged Q1 of 2022, 2023 and 2024 into one point).
        # "quarter" stays a number so the React type and chart still work; "year" and "label" are new.
        "quarterlyTurnover": [{"quarter": int(r.period[-1]), "year": int(r.period[:4]), "label": r.period,
                               "inventoryTurnover": r2(r.inventory_turnover)} for r in qtr.itertuples()],
    }


# ============================================================
# DAYS INVENTORY OUTSTANDING (DIO)
# ============================================================

@cached("dio")
def get_dio_data():
    all_data = resolve({"type": "all"})
    keys = ["dio_days", "inventory_turnover"]
    total = M.compute(keys, all_data).df.iloc[0]
    cat = M.compute(keys, all_data, dims=["category"]).df
    prod = M.compute(keys, all_data, dims=["product"]).df
    info = read_sql("SELECT ProductID, ProductName, Category FROM DimProduct")
    slow = prod.merge(info, left_on="product", right_on="ProductName").sort_values("dio_days", ascending=False).head(5)
    return {
        "daysInventoryOutstanding": f"{total.dio_days:.1f} days",
        "annualizedInventoryTurnover": f"{total.inventory_turnover:.2f}",
        "categoryDio": [{"category": r.category, "dioDays": r2(r.dio_days), "annualizedTurnover": r2(r.inventory_turnover)}
                        for r in cat.itertuples()],
        "slowestMovers": [{"productId": r.ProductID, "category": r.Category, "productName": r.ProductName,
                           "dioDays": r2(r.dio_days)} for r in slow.itertuples()],
    }


# ============================================================
# ENDPOINTS: one per dashboard, plus /all-data for Overview
# ============================================================

@app.get("/revenue")
def revenue_endpoint():
    return {"revenue": get_revenue_data()}


@app.get("/gross-margin")
def gross_margin_endpoint():
    return {"grossMargin": get_gross_margin_data()}


@app.get("/inventory-turnover")
def inventory_turnover_endpoint():
    return {"inventoryTurnover": get_inventory_turnover_data()}


@app.get("/dio")
def dio_endpoint():
    return {"dio": get_dio_data()}


@app.get("/all-data")
def get_all_data():
    return {"revenue": get_revenue_data(), "grossMargin": get_gross_margin_data(),
            "inventoryTurnover": get_inventory_turnover_data(), "dio": get_dio_data()}


# ============================================================
# RAG + AGENT
# ============================================================

class RagQueryRequest(BaseModel):
    question: str


@app.post("/rag-query")
def rag_query_endpoint(request: RagQueryRequest):
    from agent.rag import answer_from_docs
    out = answer_from_docs(request.question)
    return {"answer": out["explanation"], "found": out["found"], "sources": out["sources"]}


class AgentQueryRequest(BaseModel):
    question: str
    thread_id: str


# The agent is stateless; this backend keeps each thread's turns in memory
# (like the old MemorySaver). The Streamlit app stores history in Supabase instead.
_threads: dict[str, list] = defaultdict(list)


BAND_SERIES = ("Likely low", "Likely high")


def for_react(result: dict, question: str, history: list) -> dict:
    """
    Same answer as Streamlit, in the shape the React Ask UI renders:
      * evidence = one short line; the formulas, SQL and numbers table are sent as
        their own fields (definitions, sql, table, notes) and shown under Details
      * forecast range lines removed from the chart (the range is in the text and table),
        so the chart keeps its two lines: Actual and Forecast
      * question and turn_history kept, as the old endpoint returned them
    """
    out = dict(result)
    if result.get("definitions"):
        metrics_used = ", ".join(d.split(":")[0] for d in result["definitions"])
        out["evidence"] = f"Calculated in code from verified metric definitions: {metrics_used}. Formulas and SQL under Details."
    elif result.get("route_decision") == "adhoc" and result.get("sql"):
        out["evidence"] = "AI-written SQL, not from the verified metric catalog. Check it under Details."
    if out.get("chart"):
        chart = dict(out["chart"])
        chart["series"] = [s for s in chart.get("series", []) if s.get("name") not in BAND_SERIES]
        out["chart"] = chart
    out["question"] = question
    out["turn_history"] = [{"question": h["question"], "explanation": h["explanation"]} for h in history]
    return out


@app.post("/agent-query")
def agent_query_endpoint(request: AgentQueryRequest):
    from agent.graph import run_agent
    try:
        history = _threads[request.thread_id]
        result = run_agent(request.question, history)
        if result["route_decision"] != "limit-reached":
            history.append({"question": request.question, "explanation": result["explanation"],
                            "query": result.get("query")})
            del history[:-MAX_TURNS]
        return for_react(result, request.question, history)
    except Exception as e:
        print(f"Agent-query error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong processing that question.")


@app.get("/debug-thread/{thread_id}")
def debug_thread(thread_id: str):
    return {"turn_history": _threads.get(thread_id, [])}
