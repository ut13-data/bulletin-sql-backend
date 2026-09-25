import json
import os
import sqlite3
import re
from pydantic import BaseModel, ValidationError
from typing import TypedDict
from main import HFAPIEmbeddings

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

META_SIMILARITY_THRESHOLD = 0.3

_embeddings_model = None
_meta_example_embeddings = None

def _get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = HFAPIEmbeddings(api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return _embeddings_model

def get_meta_example_embeddings():
    global _meta_example_embeddings
    if _meta_example_embeddings is None:
        _meta_example_embeddings = _get_embeddings_model().embed_documents(META_EXAMPLES)
    return _meta_example_embeddings

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_meta_question(question: str) -> bool:
    q_embedding = _get_embeddings_model().embed_query(question)
    example_embeddings = get_meta_example_embeddings()
    best_score = max(cosine_similarity(q_embedding, ex) for ex in example_embeddings)
    print(f"DEBUG meta similarity: {best_score:.3f} for '{question}'")
    return best_score >= META_SIMILARITY_THRESHOLD

META_EXAMPLES = [
    "summarize this conversation",
    "give me a summary of what we discussed",
    "what did we talk about",
    "recap this chat",
    "how many questions have I asked",
    "what was my previous question",
    "remind me what you said earlier",
]





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
        return ToolArgs.model_validate(json.loads(tool_call.function.arguments)).model_dump()
    except (TypeError, json.JSONDecodeError, ValidationError) as e:
        print(f"DEBUG: failed to parse tool arguments: {e}")
        return {"sql": ""}

    
def safe_structured_call(messages, model, retry_note="Return ONLY valid JSON matching the required schema.", **groq_kwargs):
    call_result = safe_groq_call(model=MODEL, messages=messages, response_format={"type": "json_object"}, **groq_kwargs)
    if not call_result["ok"]:
        return None, call_result["error"]

    content = call_result["response"].choices[0].message.content
    try:
        return model.model_validate(json.loads(content)), None
    except (json.JSONDecodeError, ValidationError) as e:
        retry_messages = messages + [{"role": "user", "content": retry_note}]
        retry_result = safe_groq_call(model=MODEL, messages=retry_messages, response_format={"type": "json_object"}, **groq_kwargs)
        if not retry_result["ok"]:
            return None, retry_result["error"]
        try:
            retry_content = retry_result["response"].choices[0].message.content
            return model.model_validate(json.loads(retry_content)), None
        except (json.JSONDecodeError, ValidationError) as e2:
            return None, str(e2)

class ToolArgs(BaseModel):
    sql: str = ""

class PlanStep(BaseModel):
    agent: str
    goal: str = ""
    uses: int | None = None
    metric_hint: str | None = None

class PlannerOutput(BaseModel):
    plan: list[PlanStep] = []
    off_topic: bool = False

class ForecastData(BaseModel):
    labels: list[str]
    values: list[float]

class ChartData(BaseModel):
    style: str
    title: str
    x_axis_data: list[str]
    series: list[dict]
    summary: str = ""

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

META_EXAMPLES = [
    "summarize this conversation",
    "give me a summary of what we discussed",
    "what did we talk about",
    "recap this chat",
    "how many questions have I asked",
    "what was my previous question",
    "remind me what you said earlier",
]
META_EXAMPLE_EMBEDDINGS = None  # populated lazily on first use

def get_meta_example_embeddings():
    global META_EXAMPLE_EMBEDDINGS
    if META_EXAMPLE_EMBEDDINGS is None:
        embeddings_client = get_vectorstore().embeddings  # reuse the same embeddings object FAISS already uses
        META_EXAMPLE_EMBEDDINGS = embeddings_client.embed_documents(META_EXAMPLES)
    return META_EXAMPLE_EMBEDDINGS

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

META_SIMILARITY_THRESHOLD = 0.75  # start here, tune based on real testing




# ============================================================
# Shared recommendation generator — used by every agent path so
# recommendation is never silently empty when a real answer exists.
# ============================================================

def generate_recommendation(question: str, explanation: str) -> str:
    if not explanation or explanation.strip().lower() in ("i don't know.", "i don't know", ""):
        return ""

    prompt = (
        f"Based on this answer to a business question, write ONE short, "
        f"concrete, actionable recommendation as a COMPLETE sentence (max 20 "
        f"words, never cut off mid-thought). If genuinely no action is "
        f"warranted, respond with exactly: NONE\n\n"
        f"Question: {question}\nAnswer: {explanation}"
    )
    call_result = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
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

TURNOVER_FORMULA_NOTE = (
    "INVENTORY TURNOVER: this schema has no table linking monthly COGS "
    "directly to inventory value, so use this established proxy formula: "
    "Inventory Turnover = SUM(Sold) / AVG(ClosingStock), from "
    "FactInventoryWeekly (join DimProduct for category/name, group by month "
    "using substr(WeekEnd, 1, 7) if a monthly series is needed). Do NOT "
    "refuse an inventory turnover question by claiming a COGS-linked table "
    "is required — this proxy is the correct and only approach available "
    "in this schema.\n\n"
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
        f"{TURNOVER_FORMULA_NOTE}"
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
        return {"explanation": "I don't know.", "found": False, "sources": [], "recommendation": "", "evidence": ""}

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
        return {"explanation": "I ran into trouble answering that, please try again.", "found": False, "sources": [], "recommendation": "", "evidence": ""}

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
    evidence = f"Sources: {cited}" if (found and cited) else "No evidence found"

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
        # Normalize both sides — strip spaces/underscores so "inventory
        # turnover" (planner hint) matches "InventoryTurnover" (SQL alias).
        hint = metric_hint.lower().replace(" ", "").replace("_", "")
        for col in numeric_columns:
            col_normalized = col.lower().replace(" ", "").replace("_", "")
            if hint in col_normalized:
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
            f"{TURNOVER_FORMULA_NOTE}"
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

    chart_data, err = safe_structured_call(
        [{"role": "user", "content": chart_prompt}],
        ChartData, temperature=0, max_tokens=800, reasoning_effort="low",
    )
    if chart_data is None:
        return {
            "explanation": gathered_summary or "I gathered data but couldn't build a chart from it.",
            "found": bool(gathered_summary),
            "evidence": evidence_note,
            "recommendation": "",
            "chart": None,
        }

    explanation = chart_data.summary or gathered_summary or ""
    recommendation = generate_recommendation(question, explanation)

    return {
        "explanation": explanation,
        "found": True,
        "evidence": evidence_note,
        "recommendation": recommendation,
        "chart": {
            "style": chart_data.style,
            "title": chart_data.title,
            "x_axis_data": chart_data.x_axis_data,
            "series": chart_data.series,
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
            f"schema:\n\n{schema}\n\n{DATE_FORMAT_NOTE}{TURNOVER_FORMULA_NOTE}"
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
        parsed, err = safe_structured_call(
            [{"role": "user", "content": extract_prompt}],
            ForecastData, temperature=0, max_tokens=800,
        )
        if parsed is None or len(parsed.values) < 2 or len(parsed.values) != len(parsed.labels):
            print(f"DEBUG: forecast extraction failed: {err}")
            return {
                "explanation": gathered_summary,
                "found": True,
                "evidence": "; ".join(queries_run),
                "recommendation": "",
                "chart": None,
            }

        values = parsed.values
        labels = parsed.labels
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
        "r_squared": r_squared,
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


def get_scenario_agent_answer(
    question: str,
    history_context: str,
    input_data: list | None = None,
    metric_hint: str | None = None,
) -> dict:
    if input_data is not None:
        inferred = infer_series(input_data, metric_hint)
        if not inferred["ok"]:
            return {"explanation": inferred["error"], "found": False, "evidence": "Reused data from a previous step.", "recommendation": "", "chart": None}
        baseline = sum(inferred["values"]) if len(inferred["values"]) > 1 else inferred["values"][0]
        evidence_note = "Reused data from a previous step (no new query run)."
    else:
        schema = get_schema_summary()
        gather_prompt = (
            f"You are gathering ONE current baseline number for a what-if scenario "
            f"on Balaji Pharma data. Use run_sql_query to get it. Schema:\n\n{schema}\n\n"
            f"{DATE_FORMAT_NOTE}{TURNOVER_FORMULA_NOTE}"
            f"Once you have the number, respond in plain text with just the number "
            f"and what it represents, e.g. 'Total revenue: 4500000'."
        )
        messages = [
            {"role": "system", "content": gather_prompt},
            {"role": "user", "content": f"{history_context}Question: {question}"},
        ]
        queries_run = []
        gathered = None
        for _ in range(MAX_SQL_LOOPS):
            call_result = safe_groq_call(model=MODEL, messages=messages, tools=SQL_TOOL_SCHEMA, tool_choice="auto", temperature=0, max_tokens=500, reasoning_effort="low")
            if not call_result["ok"]:
                return {"explanation": "I ran into trouble gathering the baseline, please try again.", "found": False, "evidence": "; ".join(queries_run), "recommendation": "", "chart": None}
            message = call_result["response"].choices[0].message
            if not message.tool_calls:
                gathered = message.content or ""
                break
            tool_call = message.tool_calls[0]
            args = safe_parse_tool_args(tool_call)
            sql = args.get("sql", "")
            queries_run.append(sql)
            result = run_sql_query(sql)
            messages.append(message)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

        if gathered is None:
            return {"explanation": "I couldn't gather a baseline number for that scenario.", "found": False, "evidence": "; ".join(queries_run), "recommendation": "", "chart": None}

        extract_prompt = f"From this text, extract just the numeric baseline value:\n\n{gathered}\n\nRespond ONLY with JSON: {{\"baseline\": <number>}}"
        extract_call = safe_groq_call(model=MODEL, messages=[{"role": "user", "content": extract_prompt}], temperature=0, max_tokens=100, response_format={"type": "json_object"})
        if not extract_call["ok"]:
            return {"explanation": gathered, "found": True, "evidence": "; ".join(queries_run), "recommendation": "", "chart": None}
        try:
            baseline = float(json.loads(extract_call["response"].choices[0].message.content)["baseline"])
        except (TypeError, json.JSONDecodeError, KeyError, ValueError):
            return {"explanation": gathered, "found": True, "evidence": "; ".join(queries_run), "recommendation": "", "chart": None}
        evidence_note = "; ".join(queries_run)

    adjustment_prompt = (
        f"Extract the hypothetical percentage change being asked about in this "
        f"question. Respond ONLY with JSON: {{\"adjustment_pct\": <number, positive "
        f"for increase, negative for decrease>}}.\n\nQuestion: {question}"
    )
    adj_call = safe_groq_call(model=MODEL, messages=[{"role": "user", "content": adjustment_prompt}], temperature=0, max_tokens=100, response_format={"type": "json_object"})
    if not adj_call["ok"]:
        return {"explanation": "I couldn't work out what change you're asking about.", "found": False, "evidence": evidence_note, "recommendation": "", "chart": None}
    try:
        adjustment_pct = float(json.loads(adj_call["response"].choices[0].message.content)["adjustment_pct"])
    except (TypeError, json.JSONDecodeError, KeyError, ValueError):
        return {"explanation": "I couldn't work out what change you're asking about.", "found": False, "evidence": evidence_note, "recommendation": "", "chart": None}

    # THE ACTUAL SCENARIO MATH -- plain Python, no LLM involved.
    projected = baseline * (1 + adjustment_pct / 100)
    delta = projected - baseline

    explanation = (
        f"Current baseline: {baseline:,.0f}. A {adjustment_pct:+.1f}% change "
        f"projects to {projected:,.0f}, a difference of {delta:+,.0f}."
    )

    return {
        "explanation": explanation,
        "found": True,
        "evidence": evidence_note,
        "recommendation": (
            "This is a simple linear what-if, it doesn't account for knock-on "
            "effects elsewhere in the business -- use it to size the ballpark "
            "impact, not as a precise forecast."
        ),
        "chart": {
            "style": "bar",
            "title": "Scenario: Current vs Projected",
            "x_axis_data": ["Current", "Projected"],
            "series": [{"name": metric_hint or "Value", "values": [baseline, projected]}],
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
        "\"chart\", \"forecast\", \"scenario\". Do not invent other agent names.\n\n"
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
        "- \"what if\" hypothetical questions on a metric = sql then scenario, "
        "second step sets \"uses\" pointing at the sql step's index.\n"
        "Respond ONLY with JSON, no markdown, no backticks:\n"
        '{"plan": [...]} or {"off_topic": true}\n\n'
        "Examples:\n"
        '"What is my margin?" -> {"plan": [{"agent": "sql", "goal": "get gross margin"}]}\n'
        '"What does DistributorID represent?" -> {"plan": [{"agent": "rag", "goal": "explain DistributorID"}]}\n'
        '"How is the company performing?" -> {"plan": [{"agent": "sql", "goal": "get overall revenue and margin"}, {"agent": "rag", "goal": "explain what drives performance"}]}\n'
        '"Forecast our revenue for the next 3 months" -> {"plan": [{"agent": "sql", "goal": "get monthly revenue, last 12-18 months, ordered oldest to newest"}, {"agent": "forecast", "uses": 0, "metric_hint": "revenue"}]}\n'
        '"Plot inventory turnover by category" -> {"plan": [{"agent": "sql", "goal": "get inventory turnover by category"}, {"agent": "chart", "uses": 0, "metric_hint": "inventory turnover"}]}\n'
        '"What if we increase discount by 5%?" -> {"plan": [{"agent": "sql", "goal": "get current discount-related baseline"}, {"agent": "scenario", "uses": 0, "metric_hint": "discount"}]}\n'
        '"What is the weather today?" -> {"off_topic": true}\n\n'
        f"Question: {state['question']}"
    )


def planner_node(state: GraphState) -> GraphState:
    prompt = build_planner_prompt(state)

    parsed, err = safe_structured_call(
        [{"role": "user", "content": prompt}],
        PlannerOutput, temperature=0, max_tokens=500, reasoning_effort="low",
    )

    if parsed is None:
        state["plan"] = [{"agent": "sql", "goal": state["question"]}]
        state["route_decision"] = "sql"
        state["step_results"] = {}
        state["current_step"] = 0
        return state

    if parsed.off_topic:
        state["plan"] = []
        state["route_decision"] = "off-topic"
        state["step_results"] = {}
        state["current_step"] = 0
        return state

    VALID_AGENTS = {"sql", "rag", "chart", "forecast", "scenario"}

    plan = []
    for step in parsed.plan[:MAX_PLAN_STEPS]:
        if step.agent not in VALID_AGENTS:
            continue
        clean_step = {"agent": step.agent, "goal": step.goal or state["question"]}
        if step.uses is not None and 0 <= step.uses < len(plan):
            clean_step["uses"] = step.uses
        if step.metric_hint:
            clean_step["metric_hint"] = step.metric_hint
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
        prior_data = prior.get("data") if prior else None

        if prior_data:
            input_data = prior_data
        else:
            # The linked step ran but produced no usable data — don't let
            # this step silently fall back to its own ungrounded re-gather.
            # Report the gap honestly, same rule as the SQL agent's own
            # "say so plainly instead of guessing" instruction.
            state["step_results"][step_index] = {
                "explanation": (
                    "The previous step didn't return usable data for this "
                    "calculation, so I can't complete it."
                ),
                "found": False,
                "evidence": "",
                "recommendation": "",
                "chart": None,
            }
            state["current_step"] = step_index + 1
            return state

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
    elif agent == "scenario":
        result = get_scenario_agent_answer(
        goal, history_context, input_data=input_data, metric_hint=step.get("metric_hint")
        )
    else:
        result = {"explanation": "Unknown agent in plan.", "found": False, "evidence": "", "recommendation": ""}

    state["step_results"][step_index] = result
    state["current_step"] = step_index + 1
    return state

def meta_node(state: GraphState) -> GraphState:
    history = state.get("turn_history", [])

    if not history:
        state["explanation"] = "We haven't discussed anything yet in this conversation."
        state["found"] = True
        state["confidence"] = "N/A"
        return state

    history_text = "\n\n".join(
        f"Q{i+1}: {turn['question']}\nA{i+1}: {turn['explanation']}"
        for i, turn in enumerate(history)
    )

    prompt = (
        f"Here is the conversation history so far:\n\n{history_text}\n\n"
        f"Answer this question about the conversation itself: {state['question']}"
    )

    call_result = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
        reasoning_effort="low",
    )

    if not call_result["ok"]:
        state["explanation"] = "I ran into trouble with that, please try again."
        state["found"] = False
        return state

    state["explanation"] = call_result["response"].choices[0].message.content
    state["found"] = True
    state["confidence"] = "N/A"
    return state

def route_entry(state: GraphState) -> str:
    if is_meta_question(state["question"]):
        return "meta_node"
    return "planner_node"


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
        "scenario": "Scenario analysis",
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

def validator_node(state: GraphState) -> GraphState:
    if not state.get("found"):
        return state

    cautions = []

    for result in state.get("step_results", {}).values():
        r_squared = result.get("r_squared")
        if r_squared is not None and r_squared < 0.3:
            cautions.append(
                "the forecast trend is weak/noisy (low R\u00b2) -- treat the "
                "projected numbers as a rough estimate, not a firm prediction"
            )

    if not state.get("evidence"):
        cautions.append("this answer has no supporting data citation")

    if cautions:
        state["confidence"] = "Low - " + "; ".join(cautions)

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

graph.add_node("meta_node", meta_node)
graph.add_node("planner_node", planner_node)
graph.add_node("agent_executor_node", agent_executor_node)
graph.add_node("assemble_node", assemble_node)
graph.add_node("validator_node", validator_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("update_history_node", update_history_node)


graph.add_conditional_edges(
    START,
    route_entry,
    {
        "planner_node": "planner_node",
        "meta_node": "meta_node",
    },
)

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

graph.add_edge("assemble_node", "validator_node")
graph.add_edge("validator_node", "update_history_node")
graph.add_edge("off_topic_node", "update_history_node")
graph.add_edge("meta_node", "update_history_node")
graph.add_edge("update_history_node", END)

compiled_graph = graph.compile(checkpointer=MemorySaver())

import time

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

    start_time = time.time()
    result = compiled_graph.invoke({
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

    elapsed = time.time() - start_time
    print(f"DEBUG: run_agent took {elapsed:.2f}s for route={result.get('route_decision')}")

    return result

   

if __name__ == "__main__":
    result = run_agent("explain company structure?", thread_id="test-thread-1")
    print(result)