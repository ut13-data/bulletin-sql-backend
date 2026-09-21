import json
import os
import sqlite3
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from main import get_vectorstore, SIMILARITY_DISTANCE_THRESHOLD, client

MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 5
MAX_SQL_LOOPS = 3
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

CHART_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a read-only SQL SELECT query to get the data needed for the chart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single valid SQLite SELECT query."}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_chart",
            "description": "Submit the final chart once you have the data needed to build it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "enum": ["line", "bar", "scatter"]},
                    "title": {"type": "string"},
                    "x_axis_data": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The labels along the x-axis — e.g. categories, quarters, or product names being compared. This is ALWAYS the group/category being compared, never a metric name.",
                    },
                    "series": {
                        "type": "array",
                        "description": "One entry per METRIC being measured (e.g. 'Revenue', 'Discount %'), not one entry per category. Each series' 'values' array must have exactly one number per x_axis_data label, in the same order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The metric name, e.g. 'Discount %', not a category name.",
                                },
                                "values": {"type": "array", "items": {"type": "number"}},
                            },
                            "required": ["name", "values"],
                        },
                    },
                    "summary": {
                        "type": "string",
                        "description": "1-2 sentence plain-text explanation of what the chart shows.",
                    },
                },
                "required": ["style", "title", "x_axis_data", "series", "summary"],
            },
        },
    },
]

def get_chart_agent_answer(question: str, history_context: str) -> dict:
    schema = get_schema_summary()

    system_prompt = (
        f"You are a data visualization analyst for Balaji Pharma. First use "
        f"run_sql_query to get the data needed, then call submit_chart with the "
        f"result. Only SELECT queries work. Here is the COMPLETE database "
        f"schema — these are the ONLY tables and columns that exist:\n\n{schema}\n\n"
        f"Choose the chart style (line/bar/scatter) that best fits the question "
        f"and data shape. Always call submit_chart as your final step, never "
        f"answer in plain text.\n\n"
        f"Example: comparing discount % across 2 categories (Classical, Syrup) "
        f"with values 0.87 and 0.80 -> x_axis_data=[\"Classical\", \"Syrup\"], "
        f"series=[{{\"name\": \"Discount %\", \"values\": [0.87, 0.80]}}]. "
        f"The categories go on x_axis_data, the metric name and its values go "
        f"in ONE series entry, not one series per category."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{history_context}Question: {question}"},
    ]

    queries_run = []

    for _ in range(MAX_SQL_LOOPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=CHART_TOOL_SCHEMA,
            temperature=0,
            max_tokens=800,
            reasoning_effort="low",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # Model didn't call a tool at all — treat as failure, we always
            # need a real chart, not stray prose.
            return {
                "explanation": "I couldn't build a chart for that question.",
                "found": False,
                "evidence": "; ".join(queries_run),
                "chart": None,
            }

        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "submit_chart":
            # This is the terminal case — the model decided it's done.
            return {
                "explanation": args.get("summary", ""),
                "found": True,
                "evidence": "; ".join(queries_run),
                "chart": {
                    "style": args.get("style"),
                    "title": args.get("title"),
                    "x_axis_data": args.get("x_axis_data"),
                    "series": args.get("series"),
                },
            }

        # Otherwise it called run_sql_query — same loop pattern as sql_agent_node.
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
        "explanation": "I gathered data but couldn't finish building the chart within the query limit.",
        "found": False,
        "evidence": "; ".join(queries_run),
        "chart": None,
    }


def chart_agent_node(state: GraphState) -> GraphState:
    result = get_chart_agent_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["evidence"] = result["evidence"]
    state["chart"] = result["chart"]
    return state

def get_sql_agent_answer(question: str, history_context: str) -> dict:
    schema = get_schema_summary()

    system_prompt = (
        f"You are a SQL analyst for Balaji Pharma. Use the run_sql_query tool to "
        f"answer the question. Only SELECT queries work. Here is the COMPLETE "
        f"database schema — these are the ONLY tables and columns that exist:\n\n"
        f"{schema}\n\n"
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
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=SQL_TOOL_SCHEMA,
            temperature=0,
            max_tokens=800,
            reasoning_effort="low",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            explanation = message.content or "I don't know."
            return {
                "explanation": explanation,
                "found": bool(message.content),
                "evidence": "; ".join(queries_run),
                "recommendation": "",
            }

        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
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
        return {"explanation": "I don't know.", "found": False, "sources": []}

    relevant = relevant[:3]
    context_text = "\n\n".join(doc.page_content for doc, _ in relevant)
    prompt = (
        f"{history_context}"
        f"Context: \n\n{context_text}\n\nUsing only the context above, answer the "
        f"question. If the answer isn't in the context, say you don't know.\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        reasoning_effort="low",
    )
    answer_text = response.choices[0].message.content

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
    return {"explanation": answer_text, "found": found, "sources": sources}


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
        "(explicitly asks to see/show/plot/graph/visualize data, or compares "
        "trends over time/categories in a way clearly meant to be looked at), "
        "\"off-topic\" (unrelated, no link to prior questions).\n"
        "IMPORTANT: a vague follow-up (\"what should we do about this\", \"how "
        "can we fix it\") inherits the PREVIOUS answer's category, not a "
        "default. If the previous question was sql_agent, the follow-up is "
        "sql_agent too, unless it's clearly a different topic.\n"
        "Examples: \"What is my margin?\"->sql_agent | \"Why is margin "
        "falling?\"->sql_agent | \"What does DistributorID represent?\"->rag | "
        "\"How is the company performing?\"->both | \"Which 3 distributors had "
        "the highest revenue?\"->sql_agent | \"Show me revenue by quarter\"-> "
        "chart_agent | \"Plot inventory turnover by category\"->chart_agent | "
        "\"Visualize margin trends over the year\"->chart_agent\n"
        "One word only.\n\n"
        f"Question: {state['question']}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": classify_prompt}],
        temperature=0,
        max_tokens=100,
        reasoning_effort="low",
    )
    raw = response.choices[0].message.content.strip().lower()

    if "off-topic" in raw:
        label = "off-topic"
    elif "chart_agent" in raw:
        label = "chart_agent"
    elif "sql_agent" in raw:
        label = "sql_agent"
    elif "both" in raw:
        label = "both"
    elif "rag" in raw:
        label = "rag"
    else:
        label = "sql_agent"  # ambiguous-question fallback — dashboard app, data is more often right

    state["route_decision"] = label
    return state


def rag_node(state: GraphState) -> GraphState:
    result = get_rag_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["sources"] = result["sources"]
    return state


def sql_agent_node(state: GraphState) -> GraphState:
    result = get_sql_agent_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["evidence"] = result["evidence"]
    state["recommendation"] = result.get("recommendation", "")
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

    state["explanation"] = "\n\n".join(explanation_parts) or "I don't know."
    state["evidence"] = " | ".join(evidence_parts)
    state["recommendation"] = sql_result.get("recommendation", "")
    state["confidence"] = confidence
    state["found"] = sql_has_real_answer or rag_has_real_answer
    state["sources"] = rag_result["sources"] if rag_has_real_answer else []
    return state


def off_topic_node(state: GraphState) -> GraphState:
    state["explanation"] = "I can only answer questions about Balaji Pharma's dashboard data or business definitions."
    state["found"] = False
    state["confidence"] = "N/A"
    return state


def weak_fallback_node(state: GraphState) -> GraphState:
    state["explanation"] = "I couldn't find anything about that in Balaji Pharma's business documentation."
    state["confidence"] = "Low - not covered by available documentation"
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
    return "sql_agent_node"


def route_after_rag(state: GraphState) -> str:
    return "answer_node" if state["found"] else "weak_fallback_node"


graph = StateGraph(GraphState)

graph.add_node("classify", classify_node)
graph.add_node("rag_node", rag_node)
graph.add_node("sql_agent_node", sql_agent_node)
graph.add_node("both_node", both_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("answer_node", answer_node)
graph.add_node("weak_fallback_node", weak_fallback_node)
graph.add_node("update_history_node", update_history_node)
graph.add_node("chart_agent_node", chart_agent_node)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "rag_node": "rag_node",
        "sql_agent_node": "sql_agent_node",
        "both_node": "both_node",
        "chart_agent_node": "chart_agent_node",
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
graph.add_edge("both_node", "update_history_node")
graph.add_edge("off_topic_node", "update_history_node")
graph.add_edge("answer_node", "update_history_node")
graph.add_edge("weak_fallback_node", "update_history_node")
graph.add_edge("chart_agent_node", "update_history_node")
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
     result = get_chart_agent_answer("Show me revenue by quarter", "")
     print(result)