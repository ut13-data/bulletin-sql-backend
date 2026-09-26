"""
Executes a structured question (MetricQuery) and builds the answer.

Everything here is deterministic Python: the numbers, the sentences that
state them, the chart, and the caveats. No LLM is involved, so the stated
numbers always match the computed numbers.

Operations:
  value      one number per metric for a period (+ automatic comparison to the prior period)
  compare    the same metric across two or more periods (+ like-for-like when a period is partial)
  trend      a metric over time (monthly / quarterly / yearly)
  breakdown  a metric split by a dimension (category, product, distributor, ...)
  forecast   future months, with a backtested model choice
  scenario   what-if on price, volume, unit cost or discount
"""
import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from agent import metrics as M
from agent.formatting import fmt, fmt_change, inr
from agent.forecasting import confidence_from_mape, forecast_monthly
from agent.periods import (PeriodSpec, ResolvedPeriod, add_months, data_window, fiscal_year_of,
                           like_for_like, month_label, months_between, resolve)

Operation = Literal["value", "compare", "trend", "breakdown", "forecast", "scenario"]
Grain = Literal["month", "quarter", "fiscal_quarter", "year", "fiscal_year"]


class ScenarioSpec(BaseModel):
    driver: Literal["price", "volume", "unit_cost", "discount"]
    change: float
    change_unit: Literal["pct", "pts"] = "pct"


class MetricQuery(BaseModel):
    operation: Operation
    metrics: list[str] = Field(default_factory=lambda: ["net_revenue"])
    periods: list[PeriodSpec] = Field(default_factory=list)
    dimension: str | None = None
    filters: dict[str, list[str]] = Field(default_factory=dict)
    grain: Grain = "month"
    top_n: int | None = None
    sort: Literal["desc", "asc"] = "desc"
    horizon: int = 3
    scenario: ScenarioSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _tolerate_nulls(cls, data):
        """LLMs often send null or a bare string instead of a default or a list. Accept both."""
        if not isinstance(data, dict):
            return data
        data = {k: v for k, v in data.items() if v is not None or k in ("dimension", "top_n", "scenario")}
        if isinstance(data.get("metrics"), str):
            data["metrics"] = [data["metrics"]]
        if isinstance(data.get("periods"), dict):
            data["periods"] = [data["periods"]]
        if isinstance(data.get("filters"), dict):
            data["filters"] = {k: ([v] if isinstance(v, str) else v) for k, v in data["filters"].items()
                               if v not in (None, "", [])}
        for key in ("grain", "sort"):
            if data.get(key) == "":
                data.pop(key)
        return data


@dataclass
class Answer:
    explanation: str
    confidence: str
    chart: dict | None = None
    table: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)      # compact numbers for the recommendation writer
    found: bool = True


HIGH = "High - calculated in code from the verified metric definitions"


# ============================================================
# Helpers
# ============================================================

def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _definitions(keys: list[str]) -> list[str]:
    return [f"{M.metric(k).label}: {M.metric(k).formula}" for k in dict.fromkeys(keys)]


def _filter_phrase(filters: dict[str, list[str]]) -> str:
    if not filters:
        return ""
    parts = [f"{M.DIMENSION_LABELS[d].lower()} {' / '.join(v)}" for d, v in filters.items()]
    return " for " + " and ".join(parts)


def _period(q: MetricQuery, default: dict | None = None) -> ResolvedPeriod:
    spec = q.periods[0] if q.periods else PeriodSpec.model_validate(default or {"type": "all"})
    return resolve(spec)


def _confidence(periods: list[ResolvedPeriod]) -> str:
    partial = [p for p in periods if p.is_partial]
    if partial:
        return "Moderate - " + " ".join(p.coverage_note for p in partial)
    return HIGH


def _same_unit_metrics(keys: list[str]) -> list[str]:
    unit = M.metric(keys[0]).unit
    return [k for k in keys if M.metric(k).unit == unit]


def _chart(style: str, title: str, x: list[str], series: dict[str, list], unit: str) -> dict:
    return {
        "style": style,
        "title": title,
        "x_axis_data": [str(v) for v in x],
        "series": [{"name": n, "values": [None if _isnan(v) else round(float(v), 4) for v in vals]}
                   for n, vals in series.items()],
        "unit": unit,
    }


def _previous_comparable(p: ResolvedPeriod) -> tuple[ResolvedPeriod, ResolvedPeriod] | None:
    """
    For a single-period answer, find a fair earlier period to compare with:
      * complete period -> the period right before it (FY24 -> FY23, Nov -> Oct)
      * partial period  -> the same months one year earlier (FY25 Apr-Dec 2024 -> Apr-Dec 2023)
    Returns (current, previous) or None.
    """
    lo, _ = data_window()
    if p.kind in ("all", "range"):
        return None
    if p.is_partial:
        start, end = add_months(p.start, -12), add_months(p.end, -12)
        label = f"the same months a year earlier, {month_label(start)} to {month_label(add_months(end, -1))}"
    else:
        length = months_between(p.start, p.end)
        start, end = add_months(p.start, -length), p.start
        if start < lo:
            return None
        # Name the previous period the same way as the current one (FY24 -> FY23, 2024-Q3 -> 2024-Q2).
        value = {"fiscal_year": lambda: f"FY{fiscal_year_of(start) % 100:02d}",
                 "calendar_year": lambda: start.year,
                 "quarter": lambda: f"{start.year}-Q{(start.month - 1) // 3 + 1}",
                 "fiscal_quarter": lambda: f"FY{fiscal_year_of(start) % 100:02d}-Q{((start.month - 4) % 12) // 3 + 1}",
                 "month": lambda: start.strftime("%Y-%m")}.get(p.kind)
        label = resolve({"type": p.kind, "value": value()}).short if value else \
            f"{month_label(start)} to {month_label(add_months(end, -1))}"
    if start < lo:
        return None
    prev = ResolvedPeriod(label=label, short=label, start=start, end=end,
                          requested_start=start, requested_end=end, kind="range")
    return p, prev


# ============================================================
# value
# ============================================================

def op_value(q: MetricQuery) -> Answer:
    p = _period(q, {"type": "fiscal_year", "value": "latest_complete"})
    filters = M.resolve_filters(q.metrics, q.filters)
    res = M.compute(q.metrics, p, filters=filters)
    row = res.df.iloc[0] if len(res.df) else pd.Series(dtype=float)

    pair = _previous_comparable(p)
    prev_row, prev_sql = None, []
    if pair:
        prev_res = M.compute(q.metrics, pair[1], filters=filters)
        prev_row = prev_res.df.iloc[0] if len(prev_res.df) else None
        prev_sql = prev_res.sql

    lines, table, facts = [], [], {}
    for k in q.metrics:
        v = row.get(k, float("nan"))
        if _isnan(v):
            lines.append(f"No {M.metric(k).label.lower()} data for {p.label}{_filter_phrase(filters)}.")
            continue
        text = f"**{M.metric(k).label}** for {p.label}{_filter_phrase(filters)}: **{fmt(k, v)}**"
        if M.metric(k).unit == "inr" and abs(v) >= 1e5:
            text += f" ({fmt(k, v, exact=True)})"
        entry = {"Metric": M.metric(k).label, "Period": p.short, "Value": fmt(k, v, exact=True)}
        facts[k] = {"period": p.short, "value": fmt(k, v)}
        if prev_row is not None and not _isnan(prev_row.get(k)):
            pv = prev_row[k]
            change = fmt_change(k, pv, v)
            text += f", {change} vs {pair[1].label} ({fmt(k, pv)})"
            entry.update({"Previous": fmt(k, pv, exact=True), "Change": change})
            facts[k].update({"previous": fmt(k, pv), "change": change})
        lines.append(text + ".")
        table.append(entry)

    notes = [p.coverage_note] if p.is_partial else []
    if pair and p.is_partial:
        notes.append("Because the period is partial, the comparison uses the same months one year earlier.")

    return Answer(
        explanation="\n\n".join(lines), confidence=_confidence([p]), table=table, notes=notes,
        definitions=_definitions(q.metrics), sql=res.sql + prev_sql, facts=facts,
        found=bool(table),
    )


# ============================================================
# compare
# ============================================================

def _top_contributors(k: str, a: ResolvedPeriod, b: ResolvedPeriod, filters) -> str:
    """For additive sales metrics: which categories drove the change between two periods."""
    m = M.metric(k)
    if not m.additive or m.source != "sales" or "category" in filters:
        return ""
    da = M.compute([k], a, dims=["category"], filters=filters).df.set_index("category")[k]
    db = M.compute([k], b, dims=["category"], filters=filters).df.set_index("category")[k]
    delta = db.sub(da, fill_value=0).sort_values(key=lambda s: -s.abs())
    total = db.sum() - da.sum()
    if total == 0 or delta.empty:
        return ""
    same_dir = delta[(delta * total) > 0].head(2)
    if same_dir.empty:
        return ""
    parts = [f"{cat} ({inr(val) if m.unit == 'inr' else fmt(k, val)})" for cat, val in same_dir.items()]
    return f" The biggest contributors were {' and '.join(parts)}."


def op_compare(q: MetricQuery) -> Answer:
    if len(q.periods) < 2:
        raise M.MetricError("Please name at least two periods to compare, e.g. FY23 and FY24.")
    periods = sorted([resolve(s) for s in q.periods], key=lambda p: p.start)
    filters = M.resolve_filters(q.metrics, q.filters)

    values, sqls = {}, []
    for p in periods:
        res = M.compute(q.metrics, p, filters=filters)
        values[p.short] = res.df.iloc[0] if len(res.df) else pd.Series(dtype=float)
        sqls += res.sql

    first, last = periods[0], periods[-1]
    lines, table, facts, notes = [], [], {}, []
    for k in q.metrics:
        vals = [values[p.short].get(k) for p in periods]
        listed = ", ".join(f"{p.short} {fmt(k, v)}" for p, v in zip(periods, vals))
        change = fmt_change(k, vals[0], vals[-1])
        text = f"**{M.metric(k).label}**{_filter_phrase(filters)}: {listed}. Change {first.short} to {last.short}: **{change}**"
        if M.metric(k).unit == "inr" and not _isnan(vals[0]) and not _isnan(vals[-1]):
            text += f" ({inr(vals[-1] - vals[0])})"
        text += "."
        if len(periods) == 2 and not (first.is_partial or last.is_partial):
            text += _top_contributors(k, first, last, filters)
        lines.append(text)
        for p, v in zip(periods, vals):
            table.append({"Metric": M.metric(k).label, "Period": p.label, "Value": fmt(k, v, exact=True)})
        facts[k] = {"values": {p.short: fmt(k, v) for p, v in zip(periods, vals)}, "change": change}

    # Unequal coverage (e.g. FY25 has 9 months, FY24 has 12): add a like-for-like comparison.
    if len(periods) == 2 and first.months != last.months:
        partial, other = (last, first) if last.months < first.months else (first, last)
        lfl = like_for_like(partial, other)
        notes.append(partial.coverage_note)
        if lfl:
            lfl_res = M.compute(q.metrics, lfl, filters=filters)
            sqls += lfl_res.sql
            lfl_row = lfl_res.df.iloc[0]
            lfl_lines = []
            for k in q.metrics:
                pv, cv = lfl_row.get(k), values[partial.short].get(k)
                a, b = (pv, cv) if partial is last else (cv, pv)
                lfl_lines.append(f"{M.metric(k).label}: {fmt(k, cv)} vs {fmt(k, pv)} in {lfl.short}, "
                                 f"**{fmt_change(k, a, b)}**")
                facts[k]["like_for_like_change"] = fmt_change(k, a, b)
            lines.append(f"**Like-for-like** (same months only, because {partial.short} is partial): "
                         + "; ".join(lfl_lines) + ".")

    keys = _same_unit_metrics(q.metrics)
    chart = _chart("bar", f"{' vs '.join(M.metric(k).label for k in keys)} by period",
                   [p.short for p in periods],
                   {M.metric(k).label: [values[p.short].get(k) for p in periods] for k in keys},
                   M.metric(keys[0]).unit)

    return Answer(
        explanation="\n\n".join(lines), confidence=_confidence(periods), chart=chart, table=table,
        notes=[n for n in notes if n], definitions=_definitions(q.metrics), sql=sqls, facts=facts,
    )


# ============================================================
# trend
# ============================================================

_GRAIN_TO_PERIOD = {"year": "calendar_year", "fiscal_year": "fiscal_year",
                    "quarter": "quarter", "fiscal_quarter": "fiscal_quarter", "month": "month"}


def _bucket_is_partial(label: str, grain: str, window: ResolvedPeriod) -> bool:
    if grain == "month":
        return False
    b = resolve({"type": _GRAIN_TO_PERIOD[grain], "value": label})
    return b.is_partial or b.requested_start < window.start or b.requested_end > window.end


def op_trend(q: MetricQuery) -> Answer:
    p = _period(q)
    filters = M.resolve_filters(q.metrics, q.filters)
    res = M.compute(q.metrics, p, grain=q.grain, filters=filters)
    df = res.df
    if df.empty:
        raise M.MetricError(f"No data for {p.label}{_filter_phrase(filters)}.")

    partial = [lbl for lbl in df["period"] if _bucket_is_partial(lbl, q.grain, p)]
    x = [f"{lbl}*" if lbl in partial else lbl for lbl in df["period"]]

    grain_word = {"month": "Monthly", "quarter": "Quarterly", "fiscal_quarter": "Fiscal-quarter",
                  "year": "Yearly", "fiscal_year": "Fiscal-year"}[q.grain]
    lines, facts = [], {}
    for k in q.metrics:
        s = df.set_index("period")[k].astype(float)
        full = s.drop(index=partial, errors="ignore").dropna()
        if full.empty:
            continue
        hi_lbl, lo_lbl = full.idxmax(), full.idxmin()
        text = (f"**{grain_word} {M.metric(k).label.lower()}**{_filter_phrase(filters)}, {p.label}: "
                f"highest {hi_lbl} ({fmt(k, full[hi_lbl])}), lowest {lo_lbl} ({fmt(k, full[lo_lbl])})")
        if len(full) >= 2:
            text += f"; {full.index[0]} to {full.index[-1]}: {fmt_change(k, full.iloc[0], full.iloc[-1])}"
        text += "."
        facts[k] = {"highest": [hi_lbl, fmt(k, full[hi_lbl])], "lowest": [lo_lbl, fmt(k, full[lo_lbl])]}

        # Seasonality: with 2+ years of months, which calendar months are strongest on average?
        if q.grain == "month" and len(full) >= 24:
            by_month = full.groupby([pd.Period(i, "M").month for i in full.index]).mean()
            names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            strong = [names[i - 1] for i in by_month.sort_values(ascending=False).index[:3]]
            weak = [names[i - 1] for i in by_month.sort_values().index[:3]]
            text += f" On average the strongest months are {', '.join(strong)} and the weakest are {', '.join(weak)}."
            facts[k]["seasonality"] = {"strongest": strong, "weakest": weak}

        # Same period last year for the latest month (only meaningful for monthly data).
        if q.grain == "month" and len(full) >= 13:
            last_lbl = full.index[-1]
            ly_lbl = str(pd.Period(last_lbl, "M") - 12)
            if ly_lbl in full.index:
                yoy = fmt_change(k, full[ly_lbl], full[last_lbl])
                text += f" {last_lbl} vs {ly_lbl}: {yoy}."
                facts[k]["latest_yoy"] = yoy
        lines.append(text)

    notes = []
    if partial:
        notes.append(f"* {', '.join(partial)} {'is' if len(partial) == 1 else 'are'} partial "
                     f"(not all months have data), so {'it is' if len(partial) == 1 else 'they are'} "
                     "left out of the highest/lowest comparison.")
    if p.is_partial:
        notes.append(p.coverage_note)

    keys = _same_unit_metrics(q.metrics)
    chart = _chart("line", f"{grain_word} {', '.join(M.metric(k).label for k in keys)}", x,
                   {M.metric(k).label: df[k].tolist() for k in keys}, M.metric(keys[0]).unit)
    table = [{"Period": lbl, **{M.metric(k).label: fmt(k, r[k], exact=True) for k in q.metrics}}
             for lbl, (_, r) in zip(x, df.iterrows())]

    return Answer(explanation="\n\n".join(lines), confidence=_confidence([p]), chart=chart, table=table,
                  notes=notes, definitions=_definitions(q.metrics), sql=res.sql, facts=facts)


# ============================================================
# breakdown
# ============================================================

def op_breakdown(q: MetricQuery) -> Answer:
    if not q.dimension:
        raise M.MetricError("Please say what to split by, e.g. by category or by product.")
    p = _period(q, {"type": "fiscal_year", "value": "latest_complete"})
    filters = M.resolve_filters(q.metrics, q.filters)
    k = q.metrics[0]
    res = M.compute(q.metrics, p, dims=[q.dimension], filters=filters)
    total_res = M.compute(q.metrics, p, filters=filters)
    df = res.df.dropna(subset=[k]).sort_values(k, ascending=(q.sort == "asc")).reset_index(drop=True)
    if df.empty:
        raise M.MetricError(f"No data for {p.label}{_filter_phrase(filters)}.")

    m = M.metric(k)
    dim_label = M.DIMENSION_LABELS[q.dimension]
    total = float(total_res.df[k].iloc[0])
    n_all = len(df)
    top_n = q.top_n or (n_all if n_all <= 15 else 10)
    shown = df.head(top_n)

    if m.additive and total:
        df["share"] = df[k] / total * 100
        shown = df.head(top_n)
    lead = "Top" if q.sort == "desc" else "Bottom"
    items = []
    for _, r in shown.head(5).iterrows():
        s = f"{r[q.dimension]} {fmt(k, r[k])}"
        if "share" in df:
            s += f" ({r['share']:.1f}%)"
        items.append(s)
    text = (f"**{m.label} by {dim_label.lower()}**{_filter_phrase(filters)}, {p.label}. "
            f"{lead} {min(5, len(shown))}: {'; '.join(items)}.")
    if m.additive:
        text += f" Total: {fmt(k, total)}."
        if "share" in df and n_all > 3 and q.sort == "desc":
            text += f" The top 3 make up {df['share'].head(3).sum():.1f}% of the total."
    else:
        text += f" Overall {m.label.lower()}: {fmt(k, total)}."

    notes = []
    if k in ("gross_margin_pct", "gross_profit"):
        neg = df[df[k] < 0][q.dimension].tolist()
        if neg:
            notes.append(f"{', '.join(neg)} {'has' if len(neg) == 1 else 'have'} negative {m.label.lower()}: "
                         "average production cost is higher than the net selling price.")
    if n_all > top_n:
        rest = df.iloc[top_n:]
        note = f"Showing {top_n} of {n_all} {dim_label.lower()} values."
        if m.additive:
            note += f" The other {len(rest)} add up to {fmt(k, rest[k].sum())}."
        notes.append(note)
    if p.is_partial:
        notes.append(p.coverage_note)

    chart = _chart("bar", f"{m.label} by {dim_label.lower()}, {p.short}", shown[q.dimension].tolist(),
                   {m.label: shown[k].tolist()}, m.unit)
    table = []
    for _, r in df.iterrows():
        row = {dim_label: r[q.dimension], **{M.metric(x).label: fmt(x, r[x], exact=True) for x in q.metrics}}
        if "share" in df:
            row["Share of total"] = f"{r['share']:.1f}%"
        table.append(row)

    facts = {"metric": m.label, "period": p.short,
             "top": [[r[q.dimension], fmt(k, r[k])] for _, r in df.head(3).iterrows()],
             "bottom": [[r[q.dimension], fmt(k, r[k])] for _, r in df.tail(2).iterrows()],
             "overall": fmt(k, total)}

    return Answer(explanation=text, confidence=_confidence([p]), chart=chart, table=table, notes=notes,
                  definitions=_definitions(q.metrics), sql=res.sql + total_res.sql, facts=facts)


# ============================================================
# forecast
# ============================================================

HISTORY_MONTHS_IN_CHART = 24
SHORT_METHOD = {"holt_winters": "Holt-Winters", "seasonal_naive_growth": "same month last year x growth",
                "linear_trend": "straight line"}


def op_forecast(q: MetricQuery) -> Answer:
    k = q.metrics[0]
    m = M.metric(k)
    p = _period(q)  # history window; defaults to all data
    filters = M.resolve_filters([k], q.filters)
    res = M.compute([k], p, grain="month", filters=filters)
    hist = pd.Series(res.df[k].astype(float).values, index=res.df["period"].values).dropna()
    if len(hist) < 6:
        raise M.MetricError("Not enough monthly history to forecast (need at least 6 months).")

    f = forecast_monthly(hist, q.horizon)
    fc_months = list(f.forecast.index)
    first_fc, last_fc = pd.Period(fc_months[0], "M"), pd.Period(fc_months[-1], "M")
    span = f"{first_fc.strftime('%b %Y')}" + (f" to {last_fc.strftime('%b %Y')}" if len(fc_months) > 1 else "")

    lines = [f"**{m.label} forecast**{_filter_phrase(filters)}, {span}:"]
    rows = []
    for mo in fc_months:
        v, lo, hi = f.forecast[mo], f.low[mo], f.high[mo]
        name = pd.Period(mo, "M").strftime("%b %Y")
        lines.append(f"- {name}: **{fmt(k, v)}** (likely {fmt(k, lo)} to {fmt(k, hi)})")
        rows.append({"Month": name, "Forecast": fmt(k, v, exact=True),
                     "Likely low": fmt(k, lo, exact=True), "Likely high": fmt(k, hi, exact=True)})

    facts = {"forecast": {mo: fmt(k, v) for mo, v in f.forecast.items()}, "method": f.method_label,
             "backtest_error_pct": f.backtest_mape}

    # Compare with the same months last year when we have them (additive metrics only).
    ly = [str(pd.Period(mo, "M") - 12) for mo in fc_months]
    if m.additive and all(x in hist.index for x in ly):
        total_fc, total_ly = f.forecast.sum(), hist[ly].sum()
        change = fmt_change(k, total_ly, total_fc)
        lines.append(f"\nTotal for these months: **{fmt(k, total_fc)}**, {change} vs the same months last year "
                     f"({fmt(k, total_ly)}).")
        facts["vs_same_months_last_year"] = change

    bt = f.candidates
    method_line = f"Method: {f.method_label}, chosen because it had the smallest error when tested on recent months"
    if f.backtest_mape is not None:
        method_line += f" (average miss {f.backtest_mape:.1f}%"
        others = [f"{SHORT_METHOD[k2]} {v:.1f}%" for k2, v in bt.items() if k2 != f.method]
        if others:
            method_line += f"; other methods: {', '.join(others)}"
        method_line += ")"
    lines.append("\n" + method_line + ".")

    last_actual = hist.index[-1]
    notes = [f"The data ends in {pd.Period(last_actual, 'M').strftime('%b %Y')}, so the forecast starts from "
             f"{first_fc.strftime('%b %Y')}.",
             "A forecast assumes past patterns continue. It doesn't know about new launches, price changes "
             "or one-off events."] + f.notes

    # Chart: last 24 actual months + forecast, joined at the last actual point.
    h = hist.iloc[-HISTORY_MONTHS_IN_CHART:]
    x = list(h.index) + fc_months
    n_h = len(h)
    actual = list(h.values) + [None] * len(fc_months)
    fc_line = [None] * (n_h - 1) + [h.values[-1]] + list(f.forecast.values)
    low = [None] * (n_h - 1) + [h.values[-1]] + list(f.low.values)
    high = [None] * (n_h - 1) + [h.values[-1]] + list(f.high.values)
    chart = _chart("line", f"{m.label}: actual and forecast", x,
                   {"Actual": actual, "Forecast": fc_line, "Likely low": low, "Likely high": high}, m.unit)

    conf_level = confidence_from_mape(f.backtest_mape, len(hist))
    confidence = (f"{conf_level} - this method missed recent known months by {f.backtest_mape:.1f}% on average"
                  if f.backtest_mape is not None else f"{conf_level} - no backtest was possible")

    return Answer(explanation="\n".join(lines), confidence=confidence, chart=chart, table=rows, notes=notes,
                  definitions=_definitions([k]), sql=res.sql, facts=facts)


# ============================================================
# scenario (what-if)
# ============================================================

def op_scenario(q: MetricQuery) -> Answer:
    s = q.scenario
    if s is None:
        raise M.MetricError("Please say what changes, e.g. 'what if prices rise 5%' or 'what if unit costs rise 10%'.")
    p = _period(q, {"type": "last_n_months", "n": 12})
    keys = ["gross_sales", "discount_amount", "net_revenue", "cogs", "gross_profit", "gross_margin_pct"]
    filters = M.resolve_filters(keys, q.filters)
    res = M.compute(keys, p, filters=filters)
    b = res.df.iloc[0]
    G, D, C = float(b.gross_sales), float(b.discount_amount), float(b.cogs)
    rate = D / G if G else 0.0
    x = s.change / 100

    if s.driver == "price":
        G2, D2, C2 = G * (1 + x), D * (1 + x), C
        assumption = f"Selling prices change by {s.change:+.1f}% and the same number of units is sold."
    elif s.driver == "volume":
        G2, D2, C2 = G * (1 + x), D * (1 + x), C * (1 + x)
        assumption = f"Units sold change by {s.change:+.1f}% with the same product mix and prices."
    elif s.driver == "unit_cost":
        G2, D2, C2 = G, D, C * (1 + x)
        assumption = f"Production cost per unit changes by {s.change:+.1f}%; prices and volumes stay the same."
    else:  # discount
        new_rate = rate + x if s.change_unit == "pts" else rate * (1 + x)
        if not 0 <= new_rate < 1:
            raise M.MetricError("That discount change would take the discount rate below 0% or above 100%.")
        G2, D2, C2 = G, G * new_rate, C
        how = f"{s.change:+.1f} percentage points" if s.change_unit == "pts" else f"{s.change:+.1f}% of the current rate"
        assumption = (f"Discount rate changes by {how} ({rate * 100:.2f}% to {new_rate * 100:.2f}%); "
                      "prices and volumes stay the same.")

    before = {"net_revenue": G - D, "cogs": C, "gross_profit": G - D - C}
    after = {"net_revenue": G2 - D2, "cogs": C2, "gross_profit": G2 - D2 - C2}
    gm_before = before["gross_profit"] / before["net_revenue"] * 100 if before["net_revenue"] else float("nan")
    gm_after = after["gross_profit"] / after["net_revenue"] * 100 if after["net_revenue"] else float("nan")

    lines = [f"**What-if{_filter_phrase(filters)}, based on {p.label}.** {assumption}"]
    table = []
    for key in ("net_revenue", "cogs", "gross_profit"):
        lines.append(f"- {M.metric(key).label}: {fmt(key, before[key])} to **{fmt(key, after[key])}** "
                     f"({inr(after[key] - before[key])}, {fmt_change(key, before[key], after[key])})")
        table.append({"Metric": M.metric(key).label, "Current": fmt(key, before[key], exact=True),
                      "Scenario": fmt(key, after[key], exact=True), "Change": fmt_change(key, before[key], after[key])})
    lines.append(f"- Gross margin: {gm_before:.1f}% to **{gm_after:.1f}%** ({gm_after - gm_before:+.1f} pts)")
    table.append({"Metric": "Gross margin", "Current": f"{gm_before:.1f}%", "Scenario": f"{gm_after:.1f}%",
                  "Change": f"{gm_after - gm_before:+.1f} pts"})

    chart = _chart("bar", "Current vs scenario", ["Net revenue", "COGS", "Gross profit"],
                   {"Current": [before[k] for k in ("net_revenue", "cogs", "gross_profit")],
                    "Scenario": [after[k] for k in ("net_revenue", "cogs", "gross_profit")]}, "inr")

    notes = ["This is a simple driver model: it doesn't estimate how customers react (for example, "
             "fewer units sold after a price rise) unless you include that change yourself."]
    if p.is_partial:
        notes.append(p.coverage_note)
    facts = {"assumption": assumption,
             "gross_profit_change": inr(after["gross_profit"] - before["gross_profit"]),
             "gross_margin": f"{gm_before:.1f}% to {gm_after:.1f}%"}

    return Answer(explanation="\n".join(lines),
                  confidence="Moderate - exact arithmetic on actual data, but the scenario itself is an assumption",
                  chart=chart, table=table, notes=notes, definitions=_definitions(keys[2:6]), sql=res.sql,
                  facts=facts)


# ============================================================
# Dispatcher
# ============================================================

OPERATIONS = {"value": op_value, "compare": op_compare, "trend": op_trend,
              "breakdown": op_breakdown, "forecast": op_forecast, "scenario": op_scenario}


def run(q: MetricQuery) -> Answer:
    unknown = [k for k in q.metrics if k not in M.CATALOG]
    if unknown:
        raise M.MetricError(f"Unknown metric(s): {', '.join(unknown)}.")
    if not q.metrics:
        raise M.MetricError("No metric was named.")
    return OPERATIONS[q.operation](q)
