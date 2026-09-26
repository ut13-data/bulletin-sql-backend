"""
Question -> structured request (the ONLY job the LLM has on the main path).

The model fills in a fixed JSON form: which operation, which catalog metrics,
which periods (in PeriodSpec format), which dimension and filters. Pydantic
validates the form; code then checks metrics, periods and filter values
against the real data before anything is calculated.
"""
import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from agent import llm
from agent.config import HISTORY_TURNS_FOR_LLM
from agent.metrics import CATALOG, DIMENSION_LABELS, catalog_text, dimension_values
from agent.operations import MetricQuery
from agent.periods import available_periods_text

Intent = Literal["metric", "definition", "conversation", "adhoc", "clarify", "off_topic"]


class ParsedQuestion(BaseModel):
    intent: Intent
    standalone_question: str = ""
    query: MetricQuery | None = None
    definition_metric: str | None = None
    clarifying_question: str | None = None


# Safety net for common wordings, applied after the LLM (the LLM is asked to use catalog keys).
ALIASES = {alias: m.key for m in CATALOG.values() for alias in (m.aliases + (m.label.lower(),))}
ALIASES.update({"revenue": "net_revenue", "sales": "net_revenue", "margin": "gross_margin_pct",
                "gross_margin": "gross_margin_pct", "profit": "gross_profit", "turnover_ratio": "inventory_turnover",
                "dio": "dio_days", "discount": "discount_pct", "otd": "on_time_delivery_pct"})


def normalise_metric(key: str) -> str | None:
    k = (key or "").strip().lower()
    if k in CATALOG:
        return k
    return ALIASES.get(k) or ALIASES.get(k.replace("_", " "))


@lru_cache(maxsize=1)
def _reference_values() -> str:
    small = ["category", "distributor", "city", "outlet_type", "supplier", "material_category", "product"]
    lines = [f"- {DIMENSION_LABELS[d]} ({d}): {', '.join(dimension_values(d))}" for d in small]
    return "\n".join(lines)


SYSTEM_PROMPT = """You translate business questions about Balaji Pharma (an Ayurvedic medicine manufacturer in Indore) into a structured request. You NEVER calculate or state numbers yourself; code does all calculations.

## Data available
{periods}
"Now" means the latest month with data. Relative words refer to it: "this year" = the latest calendar year, "last year" = the one before, "this FY"/"current fiscal year" = latest fiscal year, "last quarter"/"last month" = the most recent quarter/month with data. Fiscal year runs April to March (FY24 = Apr 2023 to Mar 2024).

## Metrics (use these keys exactly)
{catalog}

## Known values (for filters; map the user's words to these)
{values}
Category words: tablets/vati -> Vati, oils/tel -> Tel, syrups -> Syrup, capsules -> Capsule, churna/powders -> Churna, arishta/asava -> Arishta, chyawanprash/kadha/classical -> Classical.

## Output: JSON only
{{
  "intent": "metric" | "definition" | "conversation" | "adhoc" | "clarify" | "off_topic",
  "standalone_question": "the question rewritten to be self-contained, using the conversation for 'it', 'that', 'same for...'",
  "query": {{                      // only when intent = "metric"
    "operation": "value" | "compare" | "trend" | "breakdown" | "forecast" | "scenario",
    "metrics": ["metric_key", ...],          // 1 to 3 keys from the list above
    "periods": [PeriodSpec, ...],            // see below; [] = default
    "dimension": null | "category" | "product" | "distributor" | "customer" | "city" | "outlet_type" | "sales_rep" | "supplier" | "material" | "material_category",
    "filters": {{"dimension": ["value", ...]}},   // e.g. {{"category": ["Churna"]}}; {{}} if none
    "grain": "month" | "quarter" | "fiscal_quarter" | "year" | "fiscal_year",
    "top_n": null | integer,
    "sort": "desc" | "asc",
    "horizon": integer,                      // forecast months ahead, default 3, max 12
    "scenario": null | {{"driver": "price" | "volume" | "unit_cost" | "discount", "change": number, "change_unit": "pct" | "pts"}}
  }},
  "definition_metric": null | "metric_key",  // intent "definition" about one catalog metric
  "clarifying_question": null | "..."        // only for intent "clarify"
}}

PeriodSpec formats:
{{"type": "fiscal_year", "value": "FY24"}}  {{"type": "fiscal_year", "value": "latest" | "previous" | "latest_complete"}}
{{"type": "calendar_year", "value": 2024}}  {{"type": "calendar_year", "value": "latest" | "previous"}}
{{"type": "quarter", "value": "2024-Q3"}}   {{"type": "fiscal_quarter", "value": "FY25-Q2"}}
{{"type": "month", "value": "2024-11"}}     {{"type": "month", "value": "latest"}}
{{"type": "last_n_months", "n": 6}}         {{"type": "range", "start": "2023-01", "end": "2023-06"}}   {{"type": "all"}}

## Rules
- "revenue", "sales" = net_revenue (after discounts). Only use gross_sales if they say gross / before discount.
- "margin" = gross_margin_pct. "profit" = gross_profit. "turnover" alone usually means revenue in Indian usage, unless they say inventory/stock turnover.
- operation:
  value = a number for a period. compare = two or more periods ("vs", "compared to", "growth", "YoY", "change from").
  trend = over time ("trend", "monthly", "by month", "over the years", "seasonality"); set grain.
  breakdown = split by a dimension ("by category", "which product", "top 5", "best/worst"); worst/lowest/bottom -> sort "asc".
  forecast = future ("forecast", "predict", "next N months", "projection"); horizon = N.
  scenario = "what if" on price, volume, unit cost or discount. "revenue drops 10%" -> volume -10.
  Discount changes: "increase discount by 2%" or "2 points" -> change_unit "pts"; "increase discount by 10% of current" -> "pct".
- Growth questions ("how much did revenue grow in FY24") -> compare the period with the one before it.
- No period mentioned: value/breakdown -> [{{"type": "fiscal_year", "value": "latest_complete"}}]; trend/forecast -> []; scenario -> [{{"type": "last_n_months", "n": 12}}].
- Asking for a period outside the data (e.g. 2026) is still intent "metric" with that period; the code will explain there is no data.
- Filters must use the known values above when possible; for products you may use a distinctive part of the name (e.g. "Triphala").
- "how is X calculated", "what does X mean" -> intent "definition" (+ definition_metric if X is a catalog metric).
- Questions about the company, its structure, processes, supply chain, tables or columns -> intent "definition" with definition_metric null.
- A data question no catalog metric can answer (e.g. about promotions, specific orders, BOM, raw material stock) -> intent "adhoc".
- Questions about this conversation itself ("summarise our chat", "what did I ask") -> intent "conversation".
- Use "clarify" only if no sensible default exists. Unrelated to Balaji Pharma -> "off_topic".
- Follow-ups: reuse the previous query and change only what the user changed (e.g. "and FY23?" -> same operation and metrics, new period).

## Examples
Q: What was revenue in FY24?
{{"intent":"metric","standalone_question":"What was net revenue in FY24?","query":{{"operation":"value","metrics":["net_revenue"],"periods":[{{"type":"fiscal_year","value":"FY24"}}],"dimension":null,"filters":{{}},"grain":"month","top_n":null,"sort":"desc","horizon":3,"scenario":null}}}}
Q: How does FY24 revenue compare to FY23?
{{"intent":"metric","standalone_question":"Compare net revenue FY24 vs FY23","query":{{"operation":"compare","metrics":["net_revenue"],"periods":[{{"type":"fiscal_year","value":"FY23"}},{{"type":"fiscal_year","value":"FY24"}}],"dimension":null,"filters":{{}},"grain":"month","top_n":null,"sort":"desc","horizon":3,"scenario":null}}}}
Q: Which 5 products had the lowest margin last year?
{{"intent":"metric","standalone_question":"Bottom 5 products by gross margin in the previous calendar year","query":{{"operation":"breakdown","metrics":["gross_margin_pct"],"periods":[{{"type":"calendar_year","value":"previous"}}],"dimension":"product","filters":{{}},"grain":"month","top_n":5,"sort":"asc","horizon":3,"scenario":null}}}}
Q: Forecast syrup sales for the next 6 months
{{"intent":"metric","standalone_question":"Forecast net revenue of the Syrup category for the next 6 months","query":{{"operation":"forecast","metrics":["net_revenue"],"periods":[],"dimension":null,"filters":{{"category":["Syrup"]}},"grain":"month","top_n":null,"sort":"desc","horizon":6,"scenario":null}}}}
Q: Show the monthly revenue trend for Chyawanprash
{{"intent":"metric","standalone_question":"Monthly net revenue trend for Chyawanprash products","query":{{"operation":"trend","metrics":["net_revenue"],"periods":[],"dimension":null,"filters":{{"product":["Chyawanprash"]}},"grain":"month","top_n":null,"sort":"desc","horizon":3,"scenario":null}}}}
Q: What happens to profit if raw material costs go up 8%?
{{"intent":"metric","standalone_question":"What if unit production cost rises 8%: effect on gross profit (last 12 months)","query":{{"operation":"scenario","metrics":["gross_profit"],"periods":[{{"type":"last_n_months","n":12}}],"dimension":null,"filters":{{}},"grain":"month","top_n":null,"sort":"desc","horizon":3,"scenario":{{"driver":"unit_cost","change":8,"change_unit":"pct"}}}}}}
Q: How is DIO calculated?
{{"intent":"definition","standalone_question":"How is days inventory outstanding calculated?","query":null,"definition_metric":"dio_days"}}
Q: Which promotions gave the biggest discount?
{{"intent":"adhoc","standalone_question":"Which promotions gave the biggest discount?","query":null}}
"""


def _history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["## Conversation so far (most recent last)"]
    for turn in history[-HISTORY_TURNS_FOR_LLM:]:
        lines.append(f"Q: {turn.get('question', '')}")
        answer = str(turn.get("explanation", ""))[:400]
        lines.append(f"A: {answer}")
        if turn.get("query"):
            lines.append(f"Query used: {json.dumps(turn['query'], default=str)}")
    return "\n".join(lines) + "\n\n"


def build_messages(question: str, history: list[dict]) -> list[dict]:
    system = SYSTEM_PROMPT.format(periods=available_periods_text(), catalog=catalog_text(),
                                  values=_reference_values())
    user = f"{_history_block(history)}Question: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean(parsed: ParsedQuestion) -> ParsedQuestion:
    """Post-validation fixes that don't need the LLM: metric aliases, bounds, dropping empty filters."""
    if parsed.intent == "metric":
        q = parsed.query
        if q is None:
            parsed.intent = "adhoc"
            return parsed
        keys = [normalise_metric(k) for k in q.metrics]
        if not q.metrics or any(k is None for k in keys):
            parsed.intent = "adhoc"      # a metric we don't define: don't guess a formula
            parsed.query = None
            return parsed
        q.metrics = list(dict.fromkeys(keys))[:3]
        q.horizon = max(1, min(q.horizon or 3, 12))
        q.filters = {d: [v for v in vals if str(v).strip()] for d, vals in (q.filters or {}).items()
                     if d in DIMENSION_LABELS and vals}
        if q.dimension and q.dimension not in DIMENSION_LABELS:
            q.dimension = None
    if parsed.definition_metric:
        parsed.definition_metric = normalise_metric(parsed.definition_metric)
    return parsed


def parse(question: str, history: list[dict]) -> tuple[ParsedQuestion | None, str | None]:
    parsed, err = llm.structured(build_messages(question, history), ParsedQuestion)
    if parsed is None:
        return None, err
    return clean(parsed), None
