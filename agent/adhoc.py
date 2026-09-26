"""
Fallback for data questions the metric catalog can't answer.

The LLM writes SQL here, so these answers are clearly labelled as ad-hoc and
get Low confidence. To keep them as consistent as possible with the catalog,
the model is given the official metric definitions and must use them.
"""
import json
import re

from agent import llm
from agent.db import read_sql, table_columns
from agent.metrics import CATALOG, PRODUCT_COST_CTE
from agent.periods import available_periods_text

MAX_SQL_LOOPS = 4
MAX_ROWS = 200
EXCLUDED_TABLES = {"fact_procurement_transactions.csv"}

SQL_TOOL = [{
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": "Run ONE read-only SQLite SELECT (or WITH ... SELECT) query against the Balaji Pharma database.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    },
}]


def _schema_text() -> str:
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in table_columns().items() if t not in EXCLUDED_TABLES)


def _definitions_text() -> str:
    return "\n".join(f"- {m.label}: {m.formula}" for m in CATALOG.values() if m.source in ("sales", "inventory"))


def safe_select(sql: str) -> dict:
    """Run model-written SQL. Single statement, SELECT/WITH only, read-only connection, row cap."""
    cleaned = sql.strip().rstrip(";").strip()
    if ";" in cleaned:
        return {"error": "Only one statement is allowed."}
    if not re.match(r"(?is)^\s*(select|with)\b", cleaned):
        return {"error": "Only SELECT queries are allowed."}
    try:
        df = read_sql(cleaned)
    except Exception as e:
        return {"error": f"SQL error: {e}"}
    rows = df.head(MAX_ROWS).to_dict(orient="records")
    return {"rows": rows, "row_count": len(df), "truncated": len(df) > MAX_ROWS}


def answer_adhoc(question: str, history_text: str = "") -> dict:
    system = (
        "You are a careful SQL analyst for Balaji Pharma. Answer using the run_sql_query tool on this "
        f"SQLite schema (these are the ONLY tables and columns):\n\n{_schema_text()}\n\n"
        f"{available_periods_text()}\n"
        "Dates are 'YYYY-MM-DD' text (sometimes with a time); filter with ranges like "
        "OrderDate >= '2023-04-01' AND OrderDate < '2024-04-01'. Fiscal year runs April to March "
        "(FY24 = Apr 2023 to Mar 2024).\n\n"
        "OFFICIAL METRIC DEFINITIONS (you must use these, never invent other formulas):\n"
        f"{_definitions_text()}\n"
        f"Production cost per product for COGS: WITH {PRODUCT_COST_CTE}\n\n"
        "Rules: never guess tables or columns that are not listed; if the data can't answer the question, "
        "say so. When you have the result, reply in 2-4 plain sentences quoting the numbers exactly as "
        "returned by the query. Do not round differently or compute new numbers in your head."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"{history_text}Question: {question}"}]
    queries, last_rows = [], None

    for _ in range(MAX_SQL_LOOPS):
        result = llm.chat(messages, tools=SQL_TOOL, tool_choice="auto", temperature=0, max_tokens=1200,
                          reasoning_effort="low")
        if not result["ok"]:
            return {"found": False, "explanation": "I ran into trouble answering that, please try again.",
                    "sql": queries, "rows": last_rows}
        msg = result["message"]
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            return {"found": bool(text) and bool(queries), "explanation": text or "I don't know.",
                    "sql": queries, "rows": last_rows}
        call = msg.tool_calls[0]
        try:
            sql = json.loads(call.function.arguments).get("sql", "")
        except (TypeError, json.JSONDecodeError):
            sql = ""
        queries.append(sql)
        out = safe_select(sql)
        last_rows = out.get("rows")
        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(out, default=str)[:12000]})

    return {"found": False, "sql": queries, "rows": last_rows,
            "explanation": "I couldn't finish this within the query limit. Try a narrower question."}
