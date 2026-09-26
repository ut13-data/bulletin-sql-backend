"""
The semantic layer: every business metric is defined ONCE here.

Chat answers, forecasts, scenarios (and the dashboards, if they import this
file) all get their numbers from `compute()`, so the same question always
gets the same number.

Rules this file enforces:
  * Ratios are always ratio-of-sums (never an average of per-row ratios).
  * Revenue means NET revenue (after discounts), unless you ask for gross sales.
  * COGS uses each product's average production cost from FactProductionBatches.
    This reconciles to FactFinanceMonthly.COGS to within 0.001% per month.
    DimProduct.UnitCost is an older standard cost that understates cost by
    roughly 37%, so it is NOT used.
  * Inventory is valued at the same production cost, so turnover and DIO are
    consistent with COGS.
  * All filter values are passed as SQL parameters, never pasted into SQL.
"""
import difflib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from agent.db import read_sql
from agent.periods import ResolvedPeriod


class MetricError(ValueError):
    """A request the metric layer can't answer. The message is shown to the user."""


# ============================================================
# 1. Sources: which table, which date column, which raw sums
# ============================================================

# Average production cost per unit, per product (the basis for COGS and inventory value).
PRODUCT_COST_CTE = """product_cost AS (
    SELECT ProductID, AVG(UnitProductionCost) AS unit_cost
    FROM FactProductionBatches
    GROUP BY ProductID
)"""
COST = "COALESCE(pc.unit_cost, p.UnitCost)"


@dataclass(frozen=True)
class Source:
    name: str
    from_sql: str
    date_col: str
    base: dict[str, str]          # raw column -> SQL aggregate
    dims: dict[str, str]          # dimension key -> SQL expression
    needs_cost_cte: bool = False


SOURCES: dict[str, Source] = {
    "sales": Source(
        name="sales",
        from_sql="""FROM FactSalesLines f
            JOIN DimProduct p ON p.ProductID = f.ProductID
            LEFT JOIN product_cost pc ON pc.ProductID = f.ProductID
            LEFT JOIN DimCustomer c ON c.CustomerID = f.CustomerID
            LEFT JOIN DimDistributor d ON d.DistributorID = f.DistributorID
            LEFT JOIN DimEmployee e ON e.EmployeeID = f.RepEmployeeID""",
        date_col="f.OrderDate",
        base={
            "gross_sales": "SUM(f.LineRevenue)",
            "discount_amount": "SUM(f.LineRevenue * f.DiscountPct)",
            "net_revenue": "SUM(f.LineRevenue * (1 - f.DiscountPct))",
            "cogs": f"SUM(f.Quantity * {COST})",
            "units_sold": "SUM(f.Quantity)",
            "orders": "COUNT(DISTINCT f.OrderID)",
            "active_customers": "COUNT(DISTINCT f.CustomerID)",
        },
        dims={
            "category": "p.Category",
            "product": "p.ProductName",
            "distributor": "d.DistributorName",
            "customer": "c.OutletName",
            "city": "c.City",
            "outlet_type": "c.OutletType",
            "sales_rep": "e.EmployeeName",
        },
        needs_cost_cte=True,
    ),
    "inventory": Source(
        name="inventory",
        from_sql="""FROM FactInventoryWeekly i
            JOIN DimProduct p ON p.ProductID = i.ProductID
            LEFT JOIN product_cost pc ON pc.ProductID = i.ProductID""",
        date_col="i.WeekEnd",
        base={
            "inv_cogs": f"SUM(i.Sold * {COST})",
            "inv_value_sum": f"SUM(i.ClosingStock * {COST})",   # summed over weeks; divided by week count later
            "inv_units_sold": "SUM(i.Sold)",
        },
        dims={"category": "p.Category", "product": "p.ProductName"},
        needs_cost_cte=True,
    ),
    "production": Source(
        name="production",
        from_sql="""FROM FactProductionBatches b
            JOIN DimProduct p ON p.ProductID = b.ProductID""",
        date_col="b.ProductionWeekEnd",
        base={
            "units_produced": "SUM(b.QuantityProduced)",
            "good_units": "SUM(b.GoodQty)",
            "rejected_units": "SUM(b.RejectQty)",
            "production_cost": "SUM(b.TotalProductionCost)",
            "batches": "COUNT(*)",
        },
        dims={"category": "p.Category", "product": "p.ProductName"},
    ),
    "procurement": Source(
        name="procurement",
        from_sql="""FROM FactPurchaseOrderLines po
            LEFT JOIN DimRawMaterial m ON m.MaterialID = po.MaterialID""",
        date_col="po.OrderDate",
        base={
            "procurement_spend": "SUM(po.TotalCost)",
            "po_lines": "COUNT(*)",
            "on_time_lines": "SUM(CASE WHEN po.Status = 'Delivered On Time' THEN 1 ELSE 0 END)",
            "late_lines": "SUM(CASE WHEN po.Status = 'Delivered On Time' THEN 0 ELSE 1 END)",
            "lead_days_sum": "SUM(julianday(substr(po.ActualDeliveryDate, 1, 10)) - julianday(substr(po.OrderDate, 1, 10)))",
            "late_days_sum": ("SUM(MAX(0, julianday(substr(po.ActualDeliveryDate, 1, 10)) "
                              "- julianday(substr(po.ExpectedDeliveryDate, 1, 10))))"),
        },
        dims={"supplier": "po.SupplierName", "material": "m.MaterialName", "material_category": "m.MaterialCategory"},
    ),
}

DIMENSION_LABELS = {
    "category": "Category", "product": "Product", "distributor": "Distributor", "customer": "Customer",
    "city": "City", "outlet_type": "Outlet type", "sales_rep": "Sales rep", "supplier": "Supplier",
    "material": "Raw material", "material_category": "Material category",
}

# ============================================================
# 2. Time grains
# ============================================================

GRAINS = ("month", "quarter", "fiscal_quarter", "year", "fiscal_year")


def time_expr(date_col: str, grain: str) -> str:
    y = f"CAST(substr({date_col}, 1, 4) AS INTEGER)"
    mo = f"CAST(substr({date_col}, 6, 2) AS INTEGER)"
    fy = f"({y} + CASE WHEN {mo} >= 4 THEN 1 ELSE 0 END)"
    return {
        "month": f"substr({date_col}, 1, 7)",
        "year": f"substr({date_col}, 1, 4)",
        "quarter": f"(substr({date_col}, 1, 4) || '-Q' || (({mo} + 2) / 3))",
        "fiscal_year": f"('FY' || substr(CAST({fy} AS TEXT), 3, 2))",
        "fiscal_quarter": f"('FY' || substr(CAST({fy} AS TEXT), 3, 2) || '-Q' || (((({mo} + 8) % 12) / 3) + 1))",
    }[grain]


# ============================================================
# 3. Metric catalog
# ============================================================

def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.astype(float).replace(0, np.nan)
    return a.astype(float) / b


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    source: str
    unit: str                      # inr | pct | ratio | days | count | units
    formula: str                   # plain-language definition shown to users
    derive: Callable[[pd.DataFrame], pd.Series]
    additive: bool                 # can be summed across products/categories (enables "share of total")
    higher_is_better: bool | None = None
    aliases: tuple = field(default_factory=tuple)


def _col(name):
    return lambda df: df[name].astype(float)


CATALOG: dict[str, Metric] = {m.key: m for m in [
    # ---- Sales ----
    Metric("net_revenue", "Net revenue", "sales", "inr",
           "Sum of LineRevenue x (1 - DiscountPct) in FactSalesLines, i.e. sales after discounts.",
           _col("net_revenue"), True, True, ("revenue", "sales", "net sales", "turnover (sales)")),
    Metric("gross_sales", "Gross sales (before discounts)", "sales", "inr",
           "Sum of LineRevenue (Quantity x UnitPrice) in FactSalesLines, before discounts.",
           _col("gross_sales"), True, True, ("gross revenue", "sales before discount")),
    Metric("discount_amount", "Discount given", "sales", "inr",
           "Sum of LineRevenue x DiscountPct in FactSalesLines.",
           _col("discount_amount"), True, False, ("discounts",)),
    Metric("discount_pct", "Discount rate", "sales", "pct",
           "Discount given / gross sales x 100.",
           lambda df: _div(df["discount_amount"], df["gross_sales"]) * 100, False, False, ("discount %",)),
    Metric("cogs", "Cost of goods sold (COGS)", "sales", "inr",
           "Units sold x each product's average production cost per unit (FactProductionBatches).",
           _col("cogs"), True, False, ("cost of goods sold", "cost of sales")),
    Metric("gross_profit", "Gross profit", "sales", "inr",
           "Net revenue - COGS.",
           lambda df: df["net_revenue"].astype(float) - df["cogs"].astype(float), True, True, ("profit",)),
    Metric("gross_margin_pct", "Gross margin", "sales", "pct",
           "Gross profit / net revenue x 100.",
           lambda df: _div(df["net_revenue"] - df["cogs"], df["net_revenue"]) * 100, False, True,
           ("margin", "gross margin %")),
    Metric("units_sold", "Units sold", "sales", "units",
           "Sum of Quantity in FactSalesLines.",
           _col("units_sold"), True, True, ("volume", "quantity sold")),
    Metric("orders", "Orders", "sales", "count",
           "Number of distinct sales orders (OrderID).",
           _col("orders"), False, True, ("order count",)),
    Metric("avg_order_value", "Average order value", "sales", "inr",
           "Net revenue / number of orders.",
           lambda df: _div(df["net_revenue"], df["orders"]), False, True, ("aov",)),
    Metric("active_customers", "Active customers", "sales", "count",
           "Distinct customers (outlets) with at least one order in the period.",
           _col("active_customers"), False, True, ("customers", "outlets")),
    Metric("revenue_per_customer", "Net revenue per active customer", "sales", "inr",
           "Net revenue / active customers.",
           lambda df: _div(df["net_revenue"], df["active_customers"]), False, True, ("revenue per outlet",)),
    # ---- Inventory ----
    Metric("inventory_turnover", "Inventory turnover (annualised)", "inventory", "ratio",
           "Cost of units sold / average inventory value, scaled to a 365-day year. Both valued at "
           "average production cost, from FactInventoryWeekly.",
           lambda df: _div(df["inv_cogs"], df["avg_inventory_value"]) * _div(pd.Series(365.0, index=df.index), df["weeks"] * 7),
           False, True, ("turnover ratio", "stock turnover")),
    Metric("dio_days", "Days inventory outstanding (DIO)", "inventory", "days",
           "Average inventory value / average daily cost of units sold (= 365 / annualised turnover).",
           lambda df: _div(df["avg_inventory_value"], _div(df["inv_cogs"], df["weeks"] * 7)),
           False, False, ("dio", "days of inventory", "days of stock")),
    Metric("avg_inventory_value", "Average inventory value", "inventory", "inr",
           "Average of weekly closing stock value (ClosingStock x average production cost).",
           _col("avg_inventory_value"), True, None, ("inventory value", "stock value")),
    # ---- Production ----
    Metric("units_produced", "Units produced", "production", "units",
           "Sum of QuantityProduced in FactProductionBatches (before rejects).",
           _col("units_produced"), True, True, ("production volume",)),
    Metric("reject_rate_pct", "Production reject rate", "production", "pct",
           "Rejected units / units produced x 100.",
           lambda df: _div(df["rejected_units"], df["units_produced"]) * 100, False, False, ("rejection rate", "yield loss")),
    Metric("production_cost", "Production cost", "production", "inr",
           "Sum of TotalProductionCost (raw material + labour + overhead) in FactProductionBatches.",
           _col("production_cost"), True, False, ()),
    Metric("unit_production_cost", "Production cost per good unit", "production", "inr",
           "Production cost / good units produced.",
           lambda df: _div(df["production_cost"], df["good_units"]), False, False, ("cost per unit",)),
    Metric("batches", "Production batches", "production", "count",
           "Number of production batches.", _col("batches"), True, None, ()),
    # ---- Procurement ----
    Metric("procurement_spend", "Procurement spend", "procurement", "inr",
           "Sum of TotalCost in FactPurchaseOrderLines, by order date.",
           _col("procurement_spend"), True, None, ("purchase spend", "purchases")),
    Metric("on_time_delivery_pct", "Supplier on-time delivery", "procurement", "pct",
           "Purchase order lines delivered on time / all lines x 100.",
           lambda df: _div(df["on_time_lines"], df["po_lines"]) * 100, False, True, ("otd", "on-time delivery")),
    Metric("late_deliveries", "Late deliveries", "procurement", "count",
           "Purchase order lines not delivered on time.", _col("late_lines"), True, False, ()),
    Metric("avg_days_late", "Average days late (late lines only)", "procurement", "days",
           "Days past expected delivery date, averaged over late lines.",
           lambda df: _div(df["late_days_sum"], df["late_lines"]), False, False, ("delay",)),
    Metric("avg_lead_time_days", "Average supplier lead time", "procurement", "days",
           "Actual delivery date - order date, averaged over purchase order lines.",
           lambda df: _div(df["lead_days_sum"], df["po_lines"]), False, False, ("lead time",)),
]}


def metric(key: str) -> Metric:
    if key not in CATALOG:
        raise MetricError(f"Unknown metric '{key}'.")
    return CATALOG[key]


def dims_for(metric_keys: list[str]) -> list[str]:
    """Dimensions every requested metric can be split by."""
    sets = [set(SOURCES[metric(k).source].dims) for k in metric_keys]
    common = set.intersection(*sets) if sets else set()
    return [d for d in DIMENSION_LABELS if d in common]


def catalog_text() -> str:
    """Metric list given to the LLM so it can only pick real metrics."""
    lines = []
    for m in CATALOG.values():
        dims = ", ".join(SOURCES[m.source].dims)
        lines.append(f"- {m.key}: {m.label}. {m.formula} (can split by: {dims})")
    return "\n".join(lines)


# ============================================================
# 4. Filters: match user words to real values in the data
# ============================================================

def dimension_values(dim: str) -> list[str]:
    for src in SOURCES.values():
        if dim in src.dims:
            expr = src.dims[dim]
            sql = f"WITH {PRODUCT_COST_CTE} SELECT DISTINCT {expr} AS v {src.from_sql} WHERE {expr} IS NOT NULL ORDER BY v"
            return read_sql(sql)["v"].astype(str).tolist()
    raise MetricError(f"Unknown dimension '{dim}'.")


def match_value(dim: str, raw: str) -> list[str]:
    """
    Map one user phrase to real values. Exact (case-insensitive) first, then
    'contains' (so 'chyawanprash' matches both Chyawanprash SKUs), then close spelling.
    """
    values = dimension_values(dim)
    needle = str(raw).strip().lower()
    exact = [v for v in values if v.lower() == needle]
    if exact:
        return exact
    contains = [v for v in values if needle in v.lower()]
    if contains:
        return contains
    close = difflib.get_close_matches(needle, [v.lower() for v in values], n=3, cutoff=0.75)
    return [v for v in values if v.lower() in close]


def resolve_filters(metric_keys: list[str], filters: dict[str, list[str]] | None) -> dict[str, list[str]]:
    resolved = {}
    valid_dims = dims_for(metric_keys)
    for dim, raw_values in (filters or {}).items():
        if not raw_values:
            continue
        if dim not in valid_dims:
            names = ", ".join(metric(k).label for k in metric_keys)
            raise MetricError(f"{names} can't be filtered by {DIMENSION_LABELS.get(dim, dim).lower()}. "
                              f"Available: {', '.join(DIMENSION_LABELS[d].lower() for d in valid_dims)}.")
        matched = []
        for raw in raw_values:
            hits = match_value(dim, raw)
            if not hits:
                options = dimension_values(dim)
                preview = ", ".join(options[:12]) + (" ..." if len(options) > 12 else "")
                raise MetricError(f"I couldn't find {DIMENSION_LABELS[dim].lower()} '{raw}'. Options: {preview}")
            matched.extend(h for h in hits if h not in matched)
        resolved[dim] = matched
    return resolved


# ============================================================
# 5. compute(): the only function that turns metrics into numbers
# ============================================================

@dataclass
class ComputeResult:
    df: pd.DataFrame                   # group columns + one column per metric key
    group_cols: list[str]
    metric_keys: list[str]
    sql: list[str]                     # exactly what ran, for the Evidence section


def _source_query(src: Source, period: ResolvedPeriod, dims: list[str], grain: str | None,
                  filters: dict[str, list[str]]) -> tuple[str, dict, list[str]]:
    select, group_cols = [], []
    if grain:
        select.append(f"{time_expr(src.date_col, grain)} AS period")
        group_cols.append("period")
    for d in dims:
        select.append(f"{src.dims[d]} AS {d}")
        group_cols.append(d)
    select += [f"{expr} AS {name}" for name, expr in src.base.items()]

    where = [f"{src.date_col} >= :start", f"{src.date_col} < :end"]
    params = {"start": period.start_str, "end": period.end_str}
    for i, (dim, values) in enumerate(filters.items()):
        names = [f"f{i}_{j}" for j in range(len(values))]
        where.append(f"{src.dims[dim]} IN ({', '.join(':' + n for n in names)})")
        params.update(dict(zip(names, values)))

    cte = f"WITH {PRODUCT_COST_CTE}\n" if src.needs_cost_cte else ""
    sql = (f"{cte}SELECT {', '.join(select)}\n{src.from_sql}\nWHERE {' AND '.join(where)}"
           + (f"\nGROUP BY {', '.join(group_cols)}" if group_cols else ""))
    return sql, params, group_cols


def _inventory_weeks(period: ResolvedPeriod, grain: str | None) -> tuple[pd.DataFrame | int, str]:
    """Number of weeks in the period (or per time bucket). Calendar weeks, independent of filters."""
    col = "i.WeekEnd"
    if grain:
        sql = (f"SELECT {time_expr(col, grain)} AS period, COUNT(DISTINCT {col}) AS weeks "
               f"FROM FactInventoryWeekly i WHERE {col} >= :start AND {col} < :end GROUP BY period")
    else:
        sql = f"SELECT COUNT(DISTINCT {col}) AS weeks FROM FactInventoryWeekly i WHERE {col} >= :start AND {col} < :end"
    df = read_sql(sql, {"start": period.start_str, "end": period.end_str})
    return (df if grain else int(df["weeks"].iloc[0])), sql


def compute(metric_keys: list[str], period: ResolvedPeriod, dims: list[str] | None = None,
            grain: str | None = None, filters: dict[str, list[str]] | None = None) -> ComputeResult:
    """
    Calculate metrics for a period, optionally split by dimensions and/or a time grain.
    `filters` must already be resolved with resolve_filters().
    """
    dims = list(dims or [])
    filters = filters or {}
    for k in metric_keys:
        metric(k)
    if grain and grain not in GRAINS:
        raise MetricError(f"Unknown time grain '{grain}'.")
    bad = [d for d in dims if d not in dims_for(metric_keys)]
    if bad:
        raise MetricError(f"Can't split {', '.join(metric(k).label for k in metric_keys)} by {', '.join(bad)}.")

    frames, sqls, group_cols = [], [], []
    for source_name in dict.fromkeys(metric(k).source for k in metric_keys):
        src = SOURCES[source_name]
        src_filters = {d: v for d, v in filters.items() if d in src.dims}
        sql, params, group_cols = _source_query(src, period, dims, grain, src_filters)
        df = read_sql(sql, params)
        sqls.append(sql)

        if source_name == "inventory":
            weeks, weeks_sql = _inventory_weeks(period, grain)
            sqls.append(weeks_sql)
            if grain:
                df = df.merge(weeks, on="period", how="left")
            else:
                df["weeks"] = weeks
            df["avg_inventory_value"] = _div(df["inv_value_sum"], df["weeks"])
        frames.append(df)

    out = frames[0]
    for other in frames[1:]:
        out = out.merge(other, on=group_cols, how="outer") if group_cols else pd.concat([out, other], axis=1)

    for k in metric_keys:
        out[k] = CATALOG[k].derive(out)

    if group_cols:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return ComputeResult(df=out, group_cols=group_cols, metric_keys=list(metric_keys), sql=sqls)


def value(metric_key: str, period: ResolvedPeriod, filters: dict[str, list[str]] | None = None) -> float:
    """Single number for one metric and period."""
    res = compute([metric_key], period, filters=filters)
    v = res.df[metric_key].iloc[0] if len(res.df) else np.nan
    return float(v) if pd.notna(v) else float("nan")
