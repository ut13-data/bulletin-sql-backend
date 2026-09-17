import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
MAX_TURNS = 5

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
    turn_history: list


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
    return "; ".join(dict.fromkeys(lines))  # dedupes while keeping order


def classify_node(state: GraphState) -> GraphState:
    history_context = build_history_context(state)

    classify_prompt = (
        f"{history_context}"
        "Classify this question into exactly one word: \"live\" (needs real-time "
        "Balaji Pharma dashboard numbers — revenue, margin, inventory, turnover, "
        "any specific metric or KPI, including follow-ups like \"what should I do "
        "about it\" that refer back to a previous live-data answer above), \"rag\" "
        "(asks about business definitions, processes, database structure, or what a "
        "term/table means), \"both\" (a broad or exploratory question that needs BOTH "
        "live numbers AND business context to answer well — e.g. \"how is the company "
        "performing\", or a question combining a live metric with a process/definition "
        "question), or \"off-topic\" (unrelated to Balaji Pharma's business entirely, "
        "with no connection to the previous questions above). "
        "Examples:\n"
        "\"What is my margin?\" -> live\n"
        "\"Why is margin falling?\" -> live\n"
        "\"What does DistributorID represent?\" -> rag\n"
        "\"How is the company performing?\" -> both\n"
        "\"What is the flow of this company and how's revenue doing?\" -> both\n"
        "Respond with only one word.\n\n"
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
    elif "both" in raw:
        label = "both"
    elif "live" in raw:
        label = "live"
    elif "rag" in raw:
        label = "rag"
    else:
        label = "live"

    state["route_decision"] = label
    return state


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


def get_live_answer(question: str, history_context: str) -> dict:
    conn = get_connection()
    dashboard_data = {
        "revenue": get_revenue_data(conn),
        "grossMargin": get_gross_margin_data(conn),
        "inventoryTurnover": get_inventory_turnover_data(conn),
        "dio": get_dio_data(conn),
    }
    conn.close()

    user_prompt = (
        f"{history_context}"
        f"Dashboard data:\n\n{json.dumps(dashboard_data)}\n\n"
        f"Question: {question}"
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

    explanation = parsed.get("explanation", "")
    return {
        "explanation": explanation,
        "evidence": parsed.get("evidence", ""),
        "recommendation": parsed.get("recommendation", ""),
        "confidence": parsed.get("confidence", ""),
        "found": bool(explanation),
    }


def rag_node(state: GraphState) -> GraphState:
    result = get_rag_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["found"] = result["found"]
    state["sources"] = result["sources"]
    return state


def live_node(state: GraphState) -> GraphState:
    result = get_live_answer(state["question"], build_history_context(state))
    state["explanation"] = result["explanation"]
    state["evidence"] = result["evidence"]
    state["recommendation"] = result["recommendation"]
    state["confidence"] = result["confidence"]
    state["found"] = result["found"]
    return state


def both_node(state: GraphState) -> GraphState:
    history_context = build_history_context(state)
    rag_result = get_rag_answer(state["question"], history_context)
    live_result = get_live_answer(state["question"], history_context)

    live_explanation_lower = live_result["explanation"].lower()
    live_has_real_answer = bool(live_result["explanation"]) and (
        "does not contain" not in live_explanation_lower
        and "cannot be answered" not in live_explanation_lower
    )
    rag_has_real_answer = rag_result["found"]

    explanation_parts = []
    if live_has_real_answer:
        explanation_parts.append(f"**From live dashboard data:** {live_result['explanation']}")
    if rag_has_real_answer:
        explanation_parts.append(f"**From business documentation:** {rag_result['explanation']}")

    if not live_has_real_answer and not rag_has_real_answer:
        confidence = "Low - not covered by available data or documentation"
    elif live_has_real_answer and rag_has_real_answer:
        confidence = "Moderate - combined live data and business context, verify before acting"
    elif live_has_real_answer:
        confidence = live_result.get("confidence") or "Moderate - based on live data only"
    else:
        confidence = "Moderate - based on business documentation only"

    evidence_parts = []
    if live_result.get("evidence"):
        evidence_parts.append(live_result["evidence"])
    if rag_has_real_answer:
        cited = format_sources(rag_result["sources"])
        if cited:
            evidence_parts.append(f"Sources: {cited}")

    state["explanation"] = "\n\n".join(explanation_parts) or "I don't know."
    state["evidence"] = " | ".join(evidence_parts)
    state["recommendation"] = live_result.get("recommendation", "")
    state["confidence"] = confidence
    state["found"] = live_has_real_answer or rag_has_real_answer
    state["sources"] = rag_result["sources"] if rag_has_real_answer else []
    return state


def off_topic_node(state: GraphState) -> GraphState:
    state["explanation"] = "I can only answer questions about Balaji Pharma's dashboard data or business definitions."
    state["found"] = False
    state["confidence"] = "N/A"
    return state


def live_weak_fallback_node(state: GraphState) -> GraphState:
    if not state.get("explanation"):
        state["explanation"] = (
            "The live dashboard data doesn't contain enough detail to confidently "
            "answer that question."
        )
    state["confidence"] = state.get("confidence") or "Low - insufficient live data"
    return state


def weak_fallback_node(state: GraphState) -> GraphState:
    state["explanation"] = "I couldn't find anything about that in Balaji Pharma's business documentation."
    state["confidence"] = "Low - not covered by available documentation"
    return state


def answer_node(state: GraphState) -> GraphState:
    if not state["confidence"]:
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
    return "live_node"


def route_after_rag(state: GraphState) -> str:
    return "answer_node" if state["found"] else "weak_fallback_node"


def route_after_live(state: GraphState) -> str:
    confidence = state.get("confidence", "")
    is_weak = (
        not state.get("explanation")
        or not state.get("found", False)
        or confidence.startswith("Low")
    )
    return "live_weak_fallback_node" if is_weak else "answer_node"


graph = StateGraph(GraphState)

graph.add_node("classify", classify_node)
graph.add_node("rag_node", rag_node)
graph.add_node("live_node", live_node)
graph.add_node("both_node", both_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("answer_node", answer_node)
graph.add_node("weak_fallback_node", weak_fallback_node)
graph.add_node("live_weak_fallback_node", live_weak_fallback_node)
graph.add_node("update_history_node", update_history_node)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "rag_node": "rag_node",
        "live_node": "live_node",
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

graph.add_conditional_edges(
    "live_node",
    route_after_live,
    {
        "answer_node": "answer_node",
        "live_weak_fallback_node": "live_weak_fallback_node",
    },
)

graph.add_edge("both_node", "update_history_node")
graph.add_edge("off_topic_node", "update_history_node")
graph.add_edge("answer_node", "update_history_node")
graph.add_edge("weak_fallback_node", "update_history_node")
graph.add_edge("live_weak_fallback_node", "update_history_node")
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
    }, config=config)


if __name__ == "__main__":
    result = run_agent("How is the company performing?", thread_id="test-both-1")
    print(result)