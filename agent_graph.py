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
    # A model-generated tool call can occasionally have malformed JSON
    # arguments. Isolating the parse here means one bad tool call fails
    # gracefully instead of crashing the whole loop.
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
            }

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

    return {
        "explanation": (
            "I ran several queries but couldn't fully resolve this within the "
            "query limit. You may want to rephrase or narrow the question."
        ),
        "found": False,
        "evidence": "; ".join(queries_run),
        "recommendation": "",
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

    return {"explanation": answer_text, "found": found, "sources": sources, "recommendation": recommendation}


# ============================================================
# Chart agent: gather data with the SQL tool (tool_choice="auto"),
# then a separate untooled JSON-mode call for the chart shape.
# ============================================================

def get_chart_agent_answer(question: str, history_context: str) -> dict:
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
            "evidence": "; ".join(queries_run),
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
            "evidence": "; ".join(queries_run),
            "recommendation": "",
            "chart": None,
        }

    explanation = chart_data.get("summary", gathered_summary or "")
    recommendation = generate_recommendation(question, explanation)

    return {
        "explanation": explanation,
        "found": True,
        "evidence": "; ".join(queries_run),
        "recommendation": recommendation,
        "chart": {
            "style": chart_data.get("style"),
            "title": chart_data.get("title"),
            "x_axis_data": chart_data.get("x_axis_data"),
            "series": chart_data.get("series"),
        },
    }


# ============================================================
# Forecast agent: SQL gathers history, Python/scikit-learn does
# the actual math. Groq never predicts the number itself.
# ============================================================

def get_forecast_agent_answer(question: str, history_context: str) -> dict:
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

    # THE ACTUAL FORECAST — real Python math, no LLM involved.
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(X, y)

    future_periods = 3
    future_X = np.arange(len(values), len(values) + future_periods).reshape(-1, 1)
    predictions = model.predict(future_X)
    predictions_list = [float(p) for p in predictions]  # NumPy floats aren't JSON-serializable

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
        "evidence": "; ".join(queries_run),
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
# Nodes
# ============================================================

def classify_node(state: GraphState) -> GraphState:
    history_context = build_history_context(state)

    classify_prompt = (
        f"{history_context}"
        "If the question is vague (\"it\", \"that\", \"how about\", \"what should "
        "I do\"), mentally rewrite it using the previous Q/A above before "
        "classifying. E.g. prev=\"why is margin falling\", now=\"how about it\" -> "
        "treat as \"how about margin falling\".\n"
        "Classify into one word: \"sql_agent\" (needs real Balaji Pharma numbers "
        "— revenue, margin, inventory, turnover, rankings, comparisons across "
        "distributors/suppliers/customers, any metric or KPI, or a follow-up on "
        "one), \"rag\" (definitions, processes, DB structure), \"both\" (broad "
        "question needing real numbers AND business context), \"chart_agent\" "
        "(the question contains words like \"chart\", \"graph\", \"plot\", "
        "\"visualize\", or explicitly asks to see/show trends/comparisons meant "
        "to be looked at visually — if the word \"chart\" or \"graph\" appears "
        "ANYWHERE in the question, always classify as chart_agent, regardless of "
        "sentence structure), \"forecast_agent\" (explicitly asks to forecast, "
        "predict, project, or estimate future values — words like \"forecast\", "
        "\"predict\", \"project\", \"next month/quarter\", \"what will\", "
        "\"expected to be\"), \"off-topic\" (unrelated, no link to prior "
        "questions).\n"
        "IMPORTANT: a vague follow-up (\"what should we do about this\", \"how "
        "can we fix it\") inherits the PREVIOUS answer's category, not a "
        "default. If the previous question was sql_agent, the follow-up is "
        "sql_agent too, unless it's clearly a different topic.\n"
        "Examples: \"What is my margin?\"->sql_agent | \"Why is margin "
        "falling?\"->sql_agent | \"What does DistributorID represent?\"->rag | "
        "\"How is the company performing?\"->both | \"Which 3 distributors had "
        "the highest revenue?\"->sql_agent | \"Show me revenue by quarter\"-> "
        "chart_agent | \"Plot inventory turnover by category\"->chart_agent | "
        "\"Visualize margin trends over the year\"->chart_agent | \"Forecast "
        "our revenue for the next 3 months\"->forecast_agent | \"What will "
        "inventory turnover look like next quarter?\"->forecast_agent\n"
        "One word only.\n\n"
        f"Question: {state['question']}"
    )

    call_result = safe_groq_call(
        model=MODEL,
        messages=[{"role": "user", "content": classify_prompt}],
        temperature=0,
        max_tokens=100,
        reasoning_effort="low",
    )

    if not call_result["ok"]:
        state["route_decision"] = "sql_agent"
        return state

    raw = call_result["response"].choices[0].message.content.strip().lower()

    if "off-topic" in raw:
        label = "off-topic"
    elif "chart_agent" in raw:
        label = "chart_agent"
    elif "forecast_agent" in raw:
        label = "forecast_agent"
    elif "sql_agent" in raw:
        label = "sql_agent"
    elif "both" in raw:
        label = "both"
    elif "rag" in raw:
        label = "rag"
    else:
        label = "sql_agent"

    state["route_decision"] = label
    return state


def rag_node(state: GraphState) -> GraphState:
    result = get_rag_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["sources"] = result["sources"]
    state["recommendation"] = result.get("recommendation", "")
    return state


def sql_agent_node(state: GraphState) -> GraphState:
    result = get_sql_agent_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["evidence"] = result["evidence"]
    state["recommendation"] = result.get("recommendation", "")
    return state


def chart_agent_node(state: GraphState) -> GraphState:
    result = get_chart_agent_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["evidence"] = result["evidence"]
    state["recommendation"] = result.get("recommendation", "")
    state["chart"] = result["chart"]
    return state


def forecast_agent_node(state: GraphState) -> GraphState:
    result = get_forecast_agent_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["evidence"] = result["evidence"]
    state["recommendation"] = result.get("recommendation", "")
    state["chart"] = result["chart"]
    return state


def both_node(state: GraphState) -> GraphState:
    history_context = build_history_context(state)
    sql_result = get_sql_agent_answer(state["question"], history_context)
    rag_result = get_rag_answer(state["question"], history_context)

    sql_has_real_answer = sql_result["found"]
    rag_has_real_answer = rag_result["found"]

    explanation_parts = []
    if sql_has_real_answer:
        explanation_parts.append(f"**From live database query:** {sql_result['explanation']}")
    if rag_has_real_answer:
        explanation_parts.append(f"**From business documentation:** {rag_result['explanation']}")

    if not sql_has_real_answer and not rag_has_real_answer:
        confidence = "Low - not covered by available data or documentation"
    elif sql_has_real_answer and rag_has_real_answer:
        confidence = "Moderate - combined query results and business context, verify before acting"
    elif sql_has_real_answer:
        confidence = "Moderate - based on database query results only"
    else:
        confidence = "Moderate - based on business documentation only"

    evidence_parts = []
    if sql_result.get("evidence"):
        evidence_parts.append(f"Query: {sql_result['evidence']}")
    if rag_has_real_answer:
        cited = format_sources(rag_result["sources"])
        if cited:
            evidence_parts.append(f"Sources: {cited}")

    final_explanation = "\n\n".join(explanation_parts) or "I don't know."

    # Prefer sql_result's own recommendation; fall back to rag_result's;
    # if both were empty, generate one from the combined explanation.
    recommendation = sql_result.get("recommendation") or rag_result.get("recommendation") or ""
    if not recommendation and (sql_has_real_answer or rag_has_real_answer):
        recommendation = generate_recommendation(state["question"], final_explanation)

    state["explanation"] = final_explanation
    state["evidence"] = " | ".join(evidence_parts)
    state["recommendation"] = recommendation
    state["confidence"] = confidence
    state["found"] = sql_has_real_answer or rag_has_real_answer
    state["sources"] = rag_result["sources"] if rag_has_real_answer else []
    return state


def off_topic_node(state: GraphState) -> GraphState:
    state["explanation"] = "I can only answer questions about Balaji Pharma's dashboard data or business definitions."
    state["found"] = False
    state["confidence"] = "N/A"
    state["recommendation"] = ""
    return state


def weak_fallback_node(state: GraphState) -> GraphState:
    state["explanation"] = "I couldn't find anything about that in Balaji Pharma's business documentation."
    state["confidence"] = "Low - not covered by available documentation"
    state["recommendation"] = ""
    return state


def answer_node(state: GraphState) -> GraphState:
    if not state["confidence"]:
        if state["route_decision"] == "sql_agent":
            state["confidence"] = "Moderate - based on database query results"
        else:
            state["confidence"] = "Moderate - based on business documentation"
    return state


def update_history_node(state: GraphState) -> GraphState:
    history = state.get("turn_history", [])
    history = history + [{"question": state["question"], "explanation": state["explanation"]}]
    state["turn_history"] = history[-MAX_TURNS:]
    return state


def route_after_classify(state: GraphState) -> str:
    if state["route_decision"] == "off-topic":
        return "off_topic_node"
    if state["route_decision"] == "rag":
        return "rag_node"
    if state["route_decision"] == "both":
        return "both_node"
    if state["route_decision"] == "chart_agent":
        return "chart_agent_node"
    if state["route_decision"] == "forecast_agent":
        return "forecast_agent_node"
    return "sql_agent_node"


def route_after_rag(state: GraphState) -> str:
    return "answer_node" if state["found"] else "weak_fallback_node"


graph = StateGraph(GraphState)

graph.add_node("classify", classify_node)
graph.add_node("rag_node", rag_node)
graph.add_node("sql_agent_node", sql_agent_node)
graph.add_node("chart_agent_node", chart_agent_node)
graph.add_node("forecast_agent_node", forecast_agent_node)
graph.add_node("both_node", both_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("answer_node", answer_node)
graph.add_node("weak_fallback_node", weak_fallback_node)
graph.add_node("update_history_node", update_history_node)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "rag_node": "rag_node",
        "sql_agent_node": "sql_agent_node",
        "chart_agent_node": "chart_agent_node",
        "forecast_agent_node": "forecast_agent_node",
        "both_node": "both_node",
        "off_topic_node": "off_topic_node",
    },
)

graph.add_conditional_edges(
    "rag_node",
    route_after_rag,
    {
        "answer_node": "answer_node",
        "weak_fallback_node": "weak_fallback_node",
    },
)

graph.add_edge("sql_agent_node", "answer_node")
graph.add_edge("chart_agent_node", "update_history_node")
graph.add_edge("forecast_agent_node", "update_history_node")
graph.add_edge("both_node", "update_history_node")
graph.add_edge("off_topic_node", "update_history_node")
graph.add_edge("answer_node", "update_history_node")
graph.add_edge("weak_fallback_node", "update_history_node")
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
    }, config=config)


if __name__ == "__main__":
    result = run_agent("explain company structure?", thread_id="test-thread-1")
    print(result)