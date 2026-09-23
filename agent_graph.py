import json
import os
import sqlite3
from typing import TypedDict

import numpy as np
from sklearn.linear_model import LinearRegression

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from main import get_vectorstore, SIMILARITY_DISTANCE_THRESHOLD, client

MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 5
MAX_SQL_LOOPS = 3
MAX_CHART_LOOPS = 5
MAX_FORECAST_LOOPS = 3
MAX_PLAN_STEPS = 3
DB_PATH = os.path.abspath("bulletin.db")
MAX_ROWS = 200


class GraphState(TypedDict):
    question: str
    route_decision: str
    explanation: str
    found: bool
    sources: list
    confidence: str
    evidence: str
    recommendation: str
    turn_history: list
    chart: dict | None
    plan: list
    step_results: dict
    current_step: int


# ============================================================
# Shared safety wrapper — every Groq call goes through this.
# ============================================================

def safe_groq_call(**kwargs) -> dict:
    try:
        response = client.chat.completions.create(**kwargs)
        return {"ok": True, "response": response}
    except Exception as e:
        print(f"DEBUG: Groq call failed: {e}")
        return {"ok": False, "error": str(e)}


def safe_parse_tool_args(tool_call) -> dict:
    try:
        return json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError) as e:
        print(f"DEBUG: failed to parse tool arguments: {e}")
        return {}


def build_history_context(state: GraphState) -> str:
    history = state.get("turn_history", [])
    if not history:
        return ""
    lines = []
    for i, turn in enumerate(history, start=1):
        lines.append(f"Previous Q{i}: {turn['question']}\nPrevious A{i}: {turn['explanation']}")
    return "\n\n".join(lines) + "\n\n"


def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = []
    for s in sources:
        label = s.get("subsection") or s.get("section") or ""
        lines.append(f"{s['document']} — {label}" if label else s["document"])
    return "; ".join(dict.fromkeys(lines))


# ============================================================
# Shared recommendation generator — used by every agent path so
# recommendation is never silently empty when a real answer exists.
# ============================================================

def generate_recommendation(question: str, explanation: str) -> str:
    if not explanation or explanation.strip().lower() in ("i don't know.", "i don't know", ""):
        return ""

    prompt = (
        f"Based on this answer to a business question, write ONE short, "
        f"concrete, actionable recommendation (max 20 words). If genuinely no "
        f"action is warranted, respond with exactly: NONE\n\n"
        f"Question: {question}\nAnswer: {explanation}"
    )
    call_result = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=60,
        reasoning_effort="low",
    )
    if not call_result["ok"]:
        return ""

    text = call_result["response"].choices[0].message.content.strip()
    if text.upper().startswith("NONE"):
        return ""
    return text


# ============================================================
# SQL tool: read-only, SELECT-only, row-capped
# ============================================================

def run_sql_query(sql: str) -> dict:
    cleaned = sql.strip()
    if not cleaned.lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed."}

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(cleaned)

        rows = cursor.fetchmany(MAX_ROWS)
        columns = [description[0] for description in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]

        conn.close()
        return {"rows": result, "row_count": len(result), "truncated": len(result) == MAX_ROWS}

    except sqlite3.Error as e:
        return {"error": f"SQL error: {str(e)}"}


def get_schema_summary() -> str:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    EXCLUDED_TABLES = {"fact_procurement_transactions.csv"}

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall() if row[0] not in EXCLUDED_TABLES]

    lines = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info([{table}])")
        columns = [row[1] for row in cursor.fetchall()]
        lines.append(f"{table}({', '.join(columns)})")

    conn.close()
    return "\n".join(lines)


DATE_FORMAT_NOTE = (
    "IMPORTANT DATE FORMATS: All date/datetime columns in this database "
    "(MonthYear in FactFinanceMonthly, OrderDate, WeekEnd, ProductionWeekEnd, "
    "LaunchDate, OnboardDate, PO_Date) use standard 'YYYY-MM-DD' or "
    "'YYYY-MM-DD HH:MM:SS' format. To extract the year, use substr(column, 1, 4). "
    "To extract year-month, use substr(column, 1, 7). Example — total revenue "
    "per year:\n"
    "SELECT substr(MonthYear, 1, 4) AS Year, SUM(Revenue) AS TotalRevenue "
    "FROM FactFinanceMonthly GROUP BY Year ORDER BY Year;\n\n"
    "EXCEPTION: DimDate.MonthYear uses a DIFFERENT format, 'Mon-YYYY' (e.g. "
    "'Apr-2022'), do not assume it matches the format above.\n\n"
)

SQL_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a read-only SQL SELECT query against the Balaji Pharma database to answer questions about revenue, margin, inventory, suppliers, distributors, or any other business data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single valid SQLite SELECT query."
                    }
                },
                "required": ["sql"]
            }
        }
    }
]


def get_sql_agent_answer(question: str, history_context: str) -> dict:
    schema = get_schema_summary()

    system_prompt = (
        f"You are a SQL analyst for Balaji Pharma. Use the run_sql_query tool to "
        f"answer the question. Only SELECT queries work. Here is the COMPLETE "
        f"database schema — these are the ONLY tables and columns that exist:\n\n"
        f"{schema}\n\n"
        f"{DATE_FORMAT_NOTE}"
        f"Never reference, suggest, or speculate about a table or column that is "
        f"not listed above, even hedged (e.g. do not say \"if you have a "
        f"delivery_log table\"). If answering the question well would require "
        f"data that doesn't exist in this schema, say so plainly instead of "
        f"guessing what might exist.\n"
        f"After you have enough information, respond with a final plain-text "
        f"answer, 2-4 sentences, citing real numbers from the query results. "
        f"Do not call the tool again once you have enough to answer."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{history_context}Question: {question}"},
    ]

    queries_run = []
    data = None  # holds the most recent query's raw rows, if any query ran

    for _ in range(MAX_SQL_LOOPS):
        call_result = safe_groq_call(
            model=MODEL,
            messages=messages,
            tools=SQL_TOOL_SCHEMA,
            temperature=0,
            max_tokens=800,
            reasoning_effort="low",
        )

        if not call_result["ok"]:
            return {
                "explanation": "I ran into trouble answering that, please try again.",
                "found": False,
                "evidence": "; ".join(queries_run),
                "recommendation": "",
                "data": data,
            }

        message = call_result["response"].choices[0].message

        if not message.tool_calls:
            explanation = message.content or "I don't know."
            recommendation = generate_recommendation(question, explanation) if explanation != "I don't know." else ""
            return {
                "explanation": explanation,
                "found": bool(message.content),
                "evidence": "; ".join(queries_run),
                "recommendation": recommendation,
                "data": data,
            }

        tool_call = message.tool_calls[0]
        args = safe_parse_tool_args(tool_call)
        sql = args.get("sql", "")
        queries_run.append(sql)

        result = run_sql_query(sql)
        data = result.get("rows")

        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

    return {
        "explanation": (
            "I ran several queries but couldn't fully resolve this within the "
            "query limit. You may want to rephrase or narrow the question."
        ),
        "found": False,
        "evidence": "; ".join(queries_run),
        "recommendation": "",
        "data": data,
    }


def get_rag_answer(question: str, history_context: str) -> dict:
    vectorstore = get_vectorstore()
    results_with_scores = vectorstore.similarity_search_with_score(question, k=5)

    relevant = [
        (doc, score) for doc, score in results_with_scores
        if score <= SIMILARITY_DISTANCE_THRESHOLD
    ]

    if not relevant:
        return {"explanation": "I don't know.", "found": False, "sources": [], "recommendation": ""}

    relevant = relevant[:3]
    context_text = "\n\n".join(doc.page_content for doc, _ in relevant)
    prompt = (
        f"{history_context}"
        f"Context: \n\n{context_text}\n\nUsing only the context above, answer the "
        f"question. If the answer isn't in the context, say you don't know.\n\n"
        f"Question: {question}"
    )

    call_result = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        reasoning_effort="low",
    )

    if not call_result["ok"]:
        return {"explanation": "I ran into trouble answering that, please try again.", "found": False, "sources": [], "recommendation": ""}

    answer_text = call_result["response"].choices[0].message.content

    sources = [
        {
            "document": doc.metadata.get("source", "Unknown"),
            "section": doc.metadata.get("section", ""),
            "subsection": doc.metadata.get("subsection", ""),
        }
        for doc, _ in relevant
    ]

    normalized = answer_text.strip().lower().replace("\u2019", "'")
    found = not normalized.startswith("i don't know")
    recommendation = generate_recommendation(question, answer_text) if found else ""

    cited = format_sources(sources)
    evidence = f"Sources: {cited}" if cited else ""

    return {"explanation": answer_text, "found": found, "sources": sources, "recommendation": recommendation, "evidence": evidence}


# ============================================================
# infer_series: given rows with unknown column names, figure out
# which column is the number series and which is the label series,
# by inspecting the actual values, never the column names.
# ============================================================

def infer_series(data: list[dict], metric_hint: str | None = None) -> dict:
    if not data:
        return {"ok": False, "error": "No data to work with."}

    columns = list(data[0].keys())

    numeric_columns = []
    string_columns = []
    for col in columns:
        values = [row.get(col) for row in data]
        is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        (numeric_columns if is_numeric else string_columns).append(col)

    if not numeric_columns:
        return {"ok": False, "error": "No numeric column found to forecast/chart."}

    value_col = None
    if metric_hint:
        hint = metric_hint.lower()
        for col in numeric_columns:
            if hint in col.lower():
                value_col = col
                break

    if value_col is None:
        if len(numeric_columns) == 1:
            value_col = numeric_columns[0]
        else:
            return {
                "ok": False,
                "error": (
                    f"Found multiple numeric columns ({', '.join(numeric_columns)}) "
                    f"and no matching metric to forecast. Please specify, e.g. "
                    f"'forecast revenue'."
                ),
            }

    if string_columns:
        label_col = string_columns[0]
        labels = [str(row.get(label_col)) for row in data]
    else:
        labels = [f"Period {i+1}" for i in range(len(data))]

    values = [float(row.get(value_col)) for row in data]

    return {"ok": True, "labels": labels, "values": values, "value_column": value_col}


# ============================================================
# Chart agent: gather data with the SQL tool (or reuse a prior
# step's data via input_data), then a JSON-mode call for chart shape.
# ============================================================

def get_chart_agent_answer(
    question: str,
    history_context: str,
    input_data: list | None = None,
    metric_hint: str | None = None,
) -> dict:
    if input_data is not None:
        inferred = infer_series(input_data, metric_hint)
        if not inferred["ok"]:
            return {
                "explanation": inferred["error"],
                "found": False,
                "evidence": "Reused data from a previous step.",
                "recommendation": "",
                "chart": None,
            }
        labels, values, value_col = inferred["labels"], inferred["values"], inferred["value_column"]
        gathered_summary = f"{value_col}: " + ", ".join(f"{l}={v}" for l, v in zip(labels, values))
        evidence_note = "Reused data from a previous step (no new query run)."
    else:
        schema = get_schema_summary()

        gather_system_prompt = (
            f"You are a data analyst gathering data for a chart about Balaji Pharma. "
            f"Use the run_sql_query tool to get the data needed. Only SELECT queries "
            f"work. Here is the COMPLETE database schema — these are the ONLY tables "
            f"and columns that exist:\n\n{schema}\n\n"
            f"{DATE_FORMAT_NOTE}"
            f"Once you have the data needed to answer the question, respond in plain "
            f"text summarizing the numbers you found. Do not call the tool again "
            f"once you have enough data."
        )

        messages = [
            {"role": "system", "content": gather_system_prompt},
            {"role": "user", "content": f"{history_context}Question: {question}"},
        ]

        queries_run = []
        gathered_summary = None

        for _ in range(MAX_CHART_LOOPS):
            call_result = safe_groq_call(
                model=MODEL,
                messages=messages,
                tools=SQL_TOOL_SCHEMA,
                tool_choice="auto",
                temperature=0,
                max_tokens=800,
                reasoning_effort="low",
            )

            if not call_result["ok"]:
                return {
                    "explanation": "I ran into trouble gathering that data, please try again.",
                    "found": False,
                    "evidence": "; ".join(queries_run),
                    "recommendation": "",
                    "chart": None,
                }

            message = call_result["response"].choices[0].message

            if not message.tool_calls:
                gathered_summary = message.content or ""
                break

            tool_call = message.tool_calls[0]
            args = safe_parse_tool_args(tool_call)
            sql = args.get("sql", "")
            queries_run.append(sql)

            result = run_sql_query(sql)

            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        if gathered_summary is None:
            return {
                "explanation": "I gathered data but couldn't finish within the query limit.",
                "found": False,
                "evidence": "; ".join(queries_run),
                "recommendation": "",
                "chart": None,
            }

        evidence_note = "; ".join(queries_run)

    chart_prompt = (
        f"Based on this data you gathered:\n\n{gathered_summary}\n\n"
        f"Original question: {question}\n\n"
        f"Respond ONLY with a JSON object in this exact shape, no markdown, no "
        f"backticks, no preamble:\n"
        f"{{\n"
        f'  "style": "line" or "bar" or "scatter",\n'
        f'  "title": "short chart title",\n'
        f'  "x_axis_data": ["array of category/time labels — the groups being compared, NEVER a metric name"],\n'
        f'  "series": [{{"name": "metric name like Revenue or Discount %", "values": [numbers, one per x_axis_data label, same order]}}],\n'
        f'  "summary": "1-2 sentence plain-text explanation of what the chart shows"\n'
        f"}}\n\n"
        f"Example: comparing discount % across 2 categories (Classical, Syrup) "
        f"with values 0.87 and 0.80 -> x_axis_data=[\"Classical\", \"Syrup\"], "
        f"series=[{{\"name\": \"Discount %\", \"values\": [0.87, 0.80]}}]."
    )

    chart_call = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": chart_prompt}],
        temperature=0,
        max_tokens=800,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )

    if not chart_call["ok"]:
        return {
            "explanation": gathered_summary or "I gathered data but couldn't build a chart from it.",
            "found": bool(gathered_summary),
            "evidence": evidence_note,
            "recommendation": "",
            "chart": None,
        }

    content = chart_call["response"].choices[0].message.content
    try:
        chart_data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {
            "explanation": gathered_summary or "I gathered data but couldn't build a chart from it.",
            "found": bool(gathered_summary),
            "evidence": evidence_note,
            "recommendation": "",
            "chart": None,
        }

    explanation = chart_data.get("summary", gathered_summary or "")
    recommendation = generate_recommendation(question, explanation)

    return {
        "explanation": explanation,
        "found": True,
        "evidence": evidence_note,
        "recommendation": recommendation,
        "chart": {
            "style": chart_data.get("style"),
            "title": chart_data.get("title"),
            "x_axis_data": chart_data.get("x_axis_data"),
            "series": chart_data.get("series"),
        },
    }


# ============================================================
# Forecast agent: uses input_data (from a prior plan step) when
# given, otherwise gathers its own history via SQL, same as before.
# Python/scikit-learn always does the actual math — Groq never
# predicts the number itself.
# ============================================================

def get_forecast_agent_answer(
    question: str,
    history_context: str,
    input_data: list | None = None,
    metric_hint: str | None = None,
) -> dict:
    if input_data is not None:
        inferred = infer_series(input_data, metric_hint)
        if not inferred["ok"]:
            return {
                "explanation": inferred["error"],
                "found": False,
                "evidence": "Reused data from a previous step.",
                "recommendation": "",
                "chart": None,
            }
        labels, values = inferred["labels"], inferred["values"]
        evidence_note = "Reused data from a previous step (no new query run)."
    else:
        schema = get_schema_summary()

        gather_system_prompt = (
            f"You are a data analyst gathering historical data to build a forecast "
            f"for Balaji Pharma. Use the run_sql_query tool to get a TIME-ORDERED "
            f"series of values (e.g. monthly revenue, ordered oldest to newest). "
            f"The most recent 12-18 periods is enough, you do NOT need the entire "
            f"history. Only SELECT queries work. Here is the COMPLETE database "
            f"schema:\n\n{schema}\n\n{DATE_FORMAT_NOTE}"
            f"Once you have the ordered historical values, respond in plain text "
            f"listing them clearly, e.g. 'Jan 2024: 100, Feb 2024: 120'. Do not "
            f"call the tool again once you have enough data. Do not attempt to "
            f"forecast yourself — just report the historical values."
        )

        messages = [
            {"role": "system", "content": gather_system_prompt},
            {"role": "user", "content": f"{history_context}Question: {question}"},
        ]

        queries_run = []
        gathered_summary = None

        for _ in range(MAX_FORECAST_LOOPS):
            call_result = safe_groq_call(
                model=MODEL,
                messages=messages,
                tools=SQL_TOOL_SCHEMA,
                tool_choice="auto",
                temperature=0,
                max_tokens=800,
                reasoning_effort="low",
            )

            if not call_result["ok"]:
                return {
                    "explanation": "I ran into trouble gathering that data, please try again.",
                    "found": False,
                    "evidence": "; ".join(queries_run),
                    "recommendation": "",
                    "chart": None,
                }

            message = call_result["response"].choices[0].message

            if not message.tool_calls:
                gathered_summary = message.content or ""
                break

            tool_call = message.tool_calls[0]
            args = safe_parse_tool_args(tool_call)
            sql = args.get("sql", "")
            queries_run.append(sql)
            result = run_sql_query(sql)

            messages.append(message)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

        if gathered_summary is None:
            return {
                "explanation": "I couldn't gather enough historical data to forecast.",
                "found": False,
                "evidence": "; ".join(queries_run),
                "recommendation": "",
                "chart": None,
            }

        extract_prompt = (
            f"From this data:\n\n{gathered_summary}\n\n"
            f"Respond ONLY with a JSON object: {{\"labels\": [...], \"values\": [...]}}, "
            f"labels as short strings (e.g. month names), values as plain numbers, "
            f"both arrays the same length, ordered oldest to newest. No markdown, "
            f"no backticks."
        )
        extract_call = safe_groq_call(
            model=MODEL,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        if not extract_call["ok"]:
            return {
                "explanation": gathered_summary,
                "found": True,
                "evidence": "; ".join(queries_run),
                "recommendation": "",
                "chart": None,
            }

        try:
            parsed = json.loads(extract_call["response"].choices[0].message.content)
            values = [float(v) for v in parsed["values"]]
            labels = [str(l) for l in parsed["labels"]]
            if len(values) < 2 or len(values) != len(labels):
                raise ValueError("insufficient or mismatched data")
        except (TypeError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"DEBUG: forecast extraction failed: {e}")
            return {
                "explanation": gathered_summary,
                "found": True,
                "evidence": "; ".join(queries_run),
                "recommendation": "",
                "chart": None,
            }

        evidence_note = "; ".join(queries_run)

    if len(values) < 2:
        return {
            "explanation": "Not enough historical data points to forecast (need at least 2).",
            "found": False,
            "evidence": evidence_note,
            "recommendation": "",
            "chart": None,
        }

    # THE ACTUAL FORECAST — real Python math, no LLM involved.
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(X, y)

    future_periods = 3
    future_X = np.arange(len(values), len(values) + future_periods).reshape(-1, 1)
    predictions = model.predict(future_X)
    predictions_list = [float(p) for p in predictions]

    trend = "increasing" if model.coef_[0] > 0 else "decreasing"
    r_squared = float(model.score(X, y))

    confidence_note = (
        "a strong, reliable trend" if r_squared >= 0.7
        else "a moderate trend, treat with some caution" if r_squared >= 0.4
        else "a weak trend, the historical data is noisy, treat this forecast as a rough estimate only"
    )

    explanation = (
        f"Based on {len(values)} periods of historical data, the trend is "
        f"{trend} (R\u00b2 = {r_squared:.2f}, indicating {confidence_note}). "
        f"Projected next {future_periods} periods: " +
        ", ".join(f"{v:,.0f}" for v in predictions_list)
    )

    forecast_labels = [f"Forecast {i+1}" for i in range(future_periods)]

    return {
        "explanation": explanation,
        "found": True,
        "evidence": evidence_note,
        "recommendation": (
            "Treat this as a straight-line projection based on past patterns, "
            "not a guarantee — revisit it if any major business change occurs "
            "(new product launch, market shift, large promotion)."
        ),
        "chart": {
            "style": "line",
            "title": "Historical + Forecast",
            "x_axis_data": labels + forecast_labels,
            "series": [
                {"name": "Actual", "values": values + [None] * future_periods},
                {"name": "Forecast", "values": [None] * len(values) + predictions_list},
            ],
        },
    }


# ============================================================
# Planner: one Groq call, produces an ordered plan instead of a
# single label. Closed agent vocabulary, capped step count.
# ============================================================

def build_planner_prompt(state: GraphState) -> str:
    history_context = build_history_context(state)
    return (
        f"{history_context}"
        "If the question is vague (\"it\", \"that\", \"how about\"), rewrite it "
        "using the previous Q/A above before planning.\n\n"
        "Break the question into a plan: an ordered list of steps. Each step "
        "uses exactly one agent from this closed set: \"sql\", \"rag\", "
        "\"chart\", \"forecast\". Do not invent other agent names.\n\n"
        "Each step is an object:\n"
        "{\n"
        '  "agent": "sql" | "rag" | "chart" | "forecast",\n'
        '  "goal": "short specific instruction for this step, e.g. '
        '\\"get monthly revenue, last 12 months\\"",\n'
        '  "uses": <int, optional — index of an earlier step whose ACTUAL DATA '
        'this step needs to do its math on. Only set this when one step\'s '
        'calculation depends on another\'s numbers, not just because multiple '
        'agents are in the plan>,\n'
        '  "metric_hint": "<optional — only for chart/forecast steps, the '
        'specific metric named in the question, e.g. \\"revenue\\" or '
        '\\"inventory turnover\\">"\n'
        "}\n\n"
        "Rules:\n"
        "- Max 3 steps. Most questions need exactly 1 step.\n"
        "- sql+rag together (needs live numbers AND business context) = 2 "
        "independent steps, no \"uses\" between them.\n"
        "- sql then forecast, or sql then chart = 2 steps, second step sets "
        "\"uses\" pointing at the sql step's index.\n"
        "- If the question is entirely unrelated to Balaji Pharma, respond with "
        '{"off_topic": true} instead of a plan.\n\n'
        "Respond ONLY with JSON, no markdown, no backticks:\n"
        '{"plan": [...]} or {"off_topic": true}\n\n'
        "Examples:\n"
        '"What is my margin?" -> {"plan": [{"agent": "sql", "goal": "get gross margin"}]}\n'
        '"What does DistributorID represent?" -> {"plan": [{"agent": "rag", "goal": "explain DistributorID"}]}\n'
        '"How is the company performing?" -> {"plan": [{"agent": "sql", "goal": "get overall revenue and margin"}, {"agent": "rag", "goal": "explain what drives performance"}]}\n'
        '"Forecast our revenue for the next 3 months" -> {"plan": [{"agent": "sql", "goal": "get monthly revenue, last 12-18 months, ordered oldest to newest"}, {"agent": "forecast", "uses": 0, "metric_hint": "revenue"}]}\n'
        '"Plot inventory turnover by category" -> {"plan": [{"agent": "sql", "goal": "get inventory turnover by category"}, {"agent": "chart", "uses": 0, "metric_hint": "inventory turnover"}]}\n'
        '"What is the weather today?" -> {"off_topic": true}\n\n'
        f"Question: {state['question']}"
    )


def planner_node(state: GraphState) -> GraphState:
    prompt = build_planner_prompt(state)

    call_result = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )

    if not call_result["ok"]:
        state["plan"] = [{"agent": "sql", "goal": state["question"]}]
        state["route_decision"] = "sql"
        state["step_results"] = {}
        state["current_step"] = 0
        return state

    content = call_result["response"].choices[0].message.content
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = {}

    if parsed.get("off_topic"):
        state["plan"] = []
        state["route_decision"] = "off-topic"
        state["step_results"] = {}
        state["current_step"] = 0
        return state

    raw_plan = parsed.get("plan") or []
    VALID_AGENTS = {"sql", "rag", "chart", "forecast"}

    plan = []
    for step in raw_plan[:MAX_PLAN_STEPS]:
        agent = step.get("agent")
        if agent not in VALID_AGENTS:
            continue
        clean_step = {"agent": agent, "goal": step.get("goal") or state["question"]}
        if isinstance(step.get("uses"), int) and 0 <= step["uses"] < len(plan):
            clean_step["uses"] = step["uses"]
        if step.get("metric_hint"):
            clean_step["metric_hint"] = step["metric_hint"]
        plan.append(clean_step)

    if not plan:
        plan = [{"agent": "sql", "goal": state["question"]}]

    state["plan"] = plan
    state["route_decision"] = "+".join(step["agent"] for step in plan)
    state["step_results"] = {}
    state["current_step"] = 0
    return state


# ============================================================
# Executor: walks the plan one step at a time (self-loops via
# conditional edge). No Groq call here — pure Python dispatch.
# ============================================================

def agent_executor_node(state: GraphState) -> GraphState:
    plan = state.get("plan", [])
    step_index = state.get("current_step", 0)

    if step_index >= len(plan):
        return state

    step = plan[step_index]
    agent = step["agent"]
    goal = step.get("goal", state["question"])
    history_context = build_history_context(state)

    input_data = None
    if "uses" in step:
        prior = state["step_results"].get(step["uses"])
        if prior:
            input_data = prior.get("data")

    if agent == "sql":
        result = get_sql_agent_answer(goal, history_context)
    elif agent == "rag":
        result = get_rag_answer(goal, history_context)
    elif agent == "forecast":
        result = get_forecast_agent_answer(
            goal, history_context, input_data=input_data, metric_hint=step.get("metric_hint")
        )
    elif agent == "chart":
        result = get_chart_agent_answer(
            goal, history_context, input_data=input_data, metric_hint=step.get("metric_hint")
        )
    else:
        result = {"explanation": "Unknown agent in plan.", "found": False, "evidence": "", "recommendation": ""}

    state["step_results"][step_index] = result
    state["current_step"] = step_index + 1
    return state


def route_after_executor(state: GraphState) -> str:
    if state["current_step"] < len(state.get("plan", [])):
        return "agent_executor_node"
    return "assemble_node"


# ============================================================
# Assemble: merges every step's result into the final response
# shape the frontend expects. Generalizes what both_node used to
# do for exactly 2 hardcoded agents to N planned steps.
# ============================================================

def assemble_node(state: GraphState) -> GraphState:
    plan = state.get("plan", [])
    step_results = state.get("step_results", {})

    label_map = {
        "sql": "From live database query",
        "rag": "From business documentation",
        "forecast": "Forecast",
        "chart": "Chart summary",
    }

    explanation_parts = []
    evidence_parts = []
    sources = []
    chart = None
    recommendation = ""
    found_count = 0

    for i, step in enumerate(plan):
        result = step_results.get(i, {})
        found = result.get("found", False)
        if found:
            found_count += 1
            explanation_parts.append(f"**{label_map.get(step['agent'], step['agent'])}:** {result.get('explanation', '')}")

        if result.get("evidence"):
            evidence_parts.append(result["evidence"])
        if step["agent"] == "rag" and result.get("sources"):
            sources = result["sources"]
        if result.get("chart"):
            chart = result["chart"]
        if result.get("recommendation") and not recommendation:
            recommendation = result["recommendation"]

    any_found = found_count > 0

    # Preserve the old single-RAG "I couldn't find anything" wording
    # instead of the generic "I don't know." fallback.
    if len(plan) == 1 and plan[0]["agent"] == "rag" and not any_found:
        final_explanation = "I couldn't find anything about that in Balaji Pharma's business documentation."
        confidence = "Low - not covered by available documentation"
    else:
        final_explanation = "\n\n".join(explanation_parts) or "I don't know."
        if not any_found:
            confidence = "Low - not covered by available data or documentation"
        elif found_count == len(plan):
            confidence = (
                "Moderate - based on available data, verify before acting" if len(plan) == 1
                else "Moderate - combined multiple sources, verify before acting"
            )
        else:
            confidence = "Moderate - based on partial data"

    if not recommendation and any_found:
        recommendation = generate_recommendation(state["question"], final_explanation)

    state["explanation"] = final_explanation
    state["evidence"] = " | ".join(evidence_parts)
    state["recommendation"] = recommendation
    state["confidence"] = confidence
    state["found"] = any_found
    state["sources"] = sources
    state["chart"] = chart
    return state


def off_topic_node(state: GraphState) -> GraphState:
    state["explanation"] = "I can only answer questions about Balaji Pharma's dashboard data or business definitions."
    state["found"] = False
    state["confidence"] = "N/A"
    state["recommendation"] = ""
    return state


def update_history_node(state: GraphState) -> GraphState:
    history = state.get("turn_history", [])
    history = history + [{"question": state["question"], "explanation": state["explanation"]}]
    state["turn_history"] = history[-MAX_TURNS:]
    return state


def route_after_planner(state: GraphState) -> str:
    return "off_topic_node" if state["route_decision"] == "off-topic" else "agent_executor_node"


graph = StateGraph(GraphState)

graph.add_node("planner_node", planner_node)
graph.add_node("agent_executor_node", agent_executor_node)
graph.add_node("assemble_node", assemble_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("update_history_node", update_history_node)

graph.add_edge(START, "planner_node")

graph.add_conditional_edges(
    "planner_node",
    route_after_planner,
    {
        "off_topic_node": "off_topic_node",
        "agent_executor_node": "agent_executor_node",
    },
)

graph.add_conditional_edges(
    "agent_executor_node",
    route_after_executor,
    {
        "agent_executor_node": "agent_executor_node",
        "assemble_node": "assemble_node",
    },
)

graph.add_edge("assemble_node", "update_history_node")
graph.add_edge("off_topic_node", "update_history_node")
graph.add_edge("update_history_node", END)

compiled_graph = graph.compile(checkpointer=MemorySaver())


def run_agent(question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    prior_state = compiled_graph.get_state(config)
    turn_history = prior_state.values.get("turn_history", []) if prior_state.values else []

    if len(turn_history) >= MAX_TURNS:
        return {
            "question": question,
            "route_decision": "limit-reached",
            "explanation": (
                "This conversation has reached its 5-question limit, to keep answers "
                "fast and grounded. Please close and reopen Ask bUlleTin to start a "
                "new conversation."
            ),
            "found": False,
            "sources": [],
            "confidence": "N/A",
            "evidence": "",
            "recommendation": "",
        }

    return compiled_graph.invoke({
        "question": question,
        "route_decision": "",
        "explanation": "",
        "found": False,
        "sources": [],
        "confidence": "",
        "evidence": "",
        "recommendation": "",
        "turn_history": turn_history,
        "chart": None,
        "plan": [],
        "step_results": {},
        "current_step": 0,
    }, config=config)


if __name__ == "__main__":
    result = run_agent("explain company structure?", thread_id="test-thread-1")
    print(result)