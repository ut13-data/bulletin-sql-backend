import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from main import (
    get_vectorstore,
    SIMILARITY_DISTANCE_THRESHOLD,
    client,
    get_connection,
    get_revenue_data,
    get_gross_margin_data,
    get_inventory_turnover_data,
    get_dio_data,
)

MODEL = "openai/gpt-oss-120b"

STRUCTURED_SYSTEM_PROMPT = (
    "You are a business intelligence assistant for Balaji Pharma, an Ayurvedic "
    "pharmaceutical distribution company. You answer questions using ONLY the data "
    "provided in the prompt. Never invent numbers, table names, column names, SQL "
    "queries, or database structure that are not explicitly present in the provided "
    "data. If the data doesn't contain enough detail to fully explain a root cause, "
    "say so explicitly rather than speculating with fabricated technical specifics or "
    "illustrative examples. Respond ONLY with a JSON object in this exact shape, with "
    "no markdown formatting, no backticks, no preamble:\n"
    "{\n"
    '  "explanation": "string, 2-4 sentences directly answering the question using specific numbers from the data",\n'
    '  "evidence": "string, optional, specific supporting figures if useful",\n'
    '  "recommendation": "string, optional, a concrete suggested action if relevant",\n'
    '  "confidence": "string, optional, one of: \'High - based on complete data\', \'Moderate - based on partial data\', \'Low - limited data available\'"\n'
    "}"
)


class GraphState(TypedDict):
    question: str
    route_decision: str
    explanation: str
    found: bool
    sources: list
    confidence: str
    evidence: str
    recommendation: str


def classify_node(state: GraphState) -> GraphState:
    classify_prompt = (
        "Classify this question into exactly one word: \"live\" (needs real-time "
        "Balaji Pharma dashboard numbers like revenue, margin, inventory), \"rag\" "
        "(asks about business definitions, processes, or database structure), or "
        "\"off-topic\" (unrelated to Balaji Pharma's business entirely). Respond with "
        f"only one word.\n\nQuestion: {state['question']}"
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
    elif "live" in raw:
        label = "live"
    elif "rag" in raw:
        label = "rag"
    else:
        label = "rag"  # safe fallback, same as classifyWithGroq in route.ts

    state["route_decision"] = label
    return state


def rag_node(state: GraphState) -> GraphState:
    vectorstore = get_vectorstore()
    results_with_scores = vectorstore.similarity_search_with_score(state["question"], k=5)

    relevant = [
        (doc, score) for doc, score in results_with_scores
        if score <= SIMILARITY_DISTANCE_THRESHOLD
    ]

    if not relevant:
        state["explanation"] = "I don't know."
        state["found"] = False
        state["sources"] = []
        return state

    relevant = relevant[:3]
    context_text = "\n\n".join(doc.page_content for doc, _ in relevant)
    prompt = (
        f"Context: \n\n{context_text}\n\nUsing only the context above, answer the "
        f"question. If the answer isn't in the context, say you don't know.\n\n"
        f"Question: {state['question']}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        reasoning_effort="low",
    )
    answer_text = response.choices[0].message.content

    state["sources"] = [
        {
            "document": doc.metadata.get("source", "Unknown"),
            "section": doc.metadata.get("section", ""),
            "subsection": doc.metadata.get("subsection", ""),
        }
        for doc, _ in relevant
    ]
    state["found"] = answer_text.strip().lower() not in ("i don't know.", "i don't know")
    state["explanation"] = answer_text
    return state


def live_node(state: GraphState) -> GraphState:
    conn = get_connection()
    dashboard_data = {
        "revenue": get_revenue_data(conn),
        "grossMargin": get_gross_margin_data(conn),
        "inventoryTurnover": get_inventory_turnover_data(conn),
        "dio": get_dio_data(conn),
    }
    conn.close()

    user_prompt = (
        f"Dashboard data:\n\n{json.dumps(dashboard_data)}\n\n"
        f"Question: {state['question']}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = {"explanation": content}

    state["explanation"] = parsed.get("explanation", "")
    state["evidence"] = parsed.get("evidence", "")
    state["recommendation"] = parsed.get("recommendation", "")
    state["confidence"] = parsed.get("confidence", "Moderate - based on live data")
    state["found"] = bool(parsed.get("explanation"))
    return state


def off_topic_node(state: GraphState) -> GraphState:
    state["explanation"] = "I can only answer questions about Balaji Pharma's dashboard data or business definitions."
    state["found"] = False
    state["confidence"] = "N/A"
    return state


def route_after_classify(state: GraphState) -> str:
    if state["route_decision"] == "off-topic":
        return "off_topic_node"
    if state["route_decision"] == "rag":
        return "rag_node"
    return "live_node"


def route_after_rag(state: GraphState) -> str:
    return "answer_node" if state["found"] else "weak_fallback_node"


def answer_node(state: GraphState) -> GraphState:
    state["confidence"] = "Moderate - based on business documentation"
    return state


def weak_fallback_node(state: GraphState) -> GraphState:
    state["explanation"] = "I couldn't find anything about that in Balaji Pharma's business documentation."
    state["confidence"] = "Low - not covered by available documentation"
    return state


graph = StateGraph(GraphState)

graph.add_node("classify", classify_node)
graph.add_node("rag_node", rag_node)
graph.add_node("live_node", live_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("answer_node", answer_node)
graph.add_node("weak_fallback_node", weak_fallback_node)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "rag_node": "rag_node",
        "live_node": "live_node",
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

graph.add_edge("live_node", END)
graph.add_edge("off_topic_node", END)
graph.add_edge("answer_node", END)
graph.add_edge("weak_fallback_node", END)

compiled_graph = graph.compile()


def run_agent(question: str) -> dict:
    return compiled_graph.invoke({
        "question": question,
        "route_decision": "",
        "explanation": "",
        "found": False,
        "sources": [],
        "confidence": "",
        "evidence": "",
        "recommendation": "",
    })


if __name__ == "__main__":
    result = run_agent("How is the revenue?")
    print(result)