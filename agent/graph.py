"""
Ask bUlleTin: the agent graph.

    question
       |
    parse_node         LLM fills a fixed JSON form (intent + structured query)
       |
    route by intent
       |-- metric        -> code computes everything from the metric catalog   (High confidence)
       |-- definition    -> catalog formula, or the business documents (RAG)
       |-- adhoc         -> LLM-written SQL, clearly labelled                    (Low confidence)
       |-- conversation  -> summary of this chat
       |-- clarify / off_topic
       |
    finalize_node      one recommendation sentence (numbers blocked in code) + response

Stateless: the caller passes this chat's earlier turns (`history`) on every
call, loaded from Supabase, so context survives restarts and days-later visits.
"""
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import llm
from agent.adhoc import answer_adhoc
from agent.config import MAX_TURNS
from agent.metrics import CATALOG, MetricError, SOURCES, dimension_values
from agent.operations import Answer, run as run_operation
from agent.parser import ParsedQuestion, parse
from agent.periods import PeriodError, data_window_text
from agent.rag import answer_from_docs

LIMIT_MESSAGE = f"This chat has reached its {MAX_TURNS}-question limit. Start a new chat to continue."


class State(TypedDict, total=False):
    question: str
    history: list
    parsed: Any          # ParsedQuestion | None
    parse_error: str | None
    result: dict         # the response being built


def _base(route: str, explanation: str, found: bool, confidence: str, **extra) -> dict:
    out = {
        "route_decision": route, "explanation": explanation, "found": found, "confidence": confidence,
        "evidence": "", "recommendation": "", "sources": [], "chart": None, "plan": [], "query": None,
        "table": [], "notes": [], "definitions": [], "sql": [], "data_through": data_window_text(),
    }
    out.update(extra)
    return out


# ============================================================
# Nodes
# ============================================================

def parse_node(state: State) -> State:
    parsed, err = parse(state["question"], state.get("history", []))
    state["parsed"], state["parse_error"] = parsed, err
    return state


def route(state: State) -> str:
    parsed: ParsedQuestion | None = state.get("parsed")
    if parsed is None:
        return "adhoc_node"     # couldn't structure it: fall back to the labelled ad-hoc path
    return {"metric": "metric_node", "definition": "definition_node", "adhoc": "adhoc_node",
            "conversation": "conversation_node", "clarify": "clarify_node",
            "off_topic": "off_topic_node"}[parsed.intent]


def metric_node(state: State) -> State:
    parsed: ParsedQuestion = state["parsed"]
    q = parsed.query
    try:
        ans: Answer = run_operation(q)
    except (MetricError, PeriodError) as e:
        state["result"] = _base("metric", str(e), False, "N/A", query=q.model_dump())
        return state

    evidence = "Definitions: " + " | ".join(ans.definitions)
    state["result"] = _base(
        "metric", ans.explanation, ans.found, ans.confidence,
        evidence=evidence, chart=ans.chart, table=ans.table, notes=ans.notes, definitions=ans.definitions,
        sql=ans.sql, query=q.model_dump(), facts=ans.facts,
        plan=[{"agent": "metric", "goal": f"{q.operation}: {', '.join(q.metrics)}"}],
    )
    return state


def definition_node(state: State) -> State:
    parsed: ParsedQuestion = state["parsed"]
    key = parsed.definition_metric
    if key and key in CATALOG:
        m = CATALOG[key]
        dims = ", ".join(SOURCES[m.source].dims)
        text = (f"**{m.label}**: {m.formula}\n\nIt can be broken down by {dims}, and is calculated for any period "
                f"with data ({data_window_text()}).")
        state["result"] = _base("definition", text, True, "High - from the metric catalog used for every answer",
                                evidence=f"Metric catalog: {m.key}",
                                plan=[{"agent": "catalog", "goal": f"define {m.key}"}])
        return state

    rag = answer_from_docs(parsed.standalone_question or state["question"])
    sources = rag.get("sources", [])
    cited = "; ".join(dict.fromkeys(f"{s['document']} - {s.get('subsection') or s.get('section')}" for s in sources))
    state["result"] = _base(
        "rag", rag["explanation"], rag["found"],
        "Moderate - from Balaji Pharma's business documents" if rag["found"] else "Low - not covered by the documents",
        evidence=f"Sources: {cited}" if cited else "", sources=sources,
        plan=[{"agent": "rag", "goal": "answer from business documents"}],
    )
    return state


def _history_text(history: list) -> str:
    if not history:
        return ""
    return "\n".join(f"Previous Q: {h['question']}\nPrevious A: {str(h.get('explanation', ''))[:400]}"
                     for h in history[-3:]) + "\n\n"


def adhoc_node(state: State) -> State:
    parsed: ParsedQuestion | None = state.get("parsed")
    question = (parsed.standalone_question if parsed and parsed.standalone_question else state["question"])
    out = answer_adhoc(question, _history_text(state.get("history", [])))
    label = ("**Ad-hoc query** (this question isn't covered by the verified metric catalog, so the AI wrote "
             "the SQL itself; check it under Details):\n\n")
    state["result"] = _base(
        "adhoc", label + out["explanation"], out["found"],
        "Low - AI-written SQL, not from the verified metric catalog" if out["found"] else "N/A",
        evidence="Ad-hoc SQL (see Details)", sql=out["sql"],
        table=(out.get("rows") or [])[:50], plan=[{"agent": "adhoc_sql", "goal": question}],
    )
    return state


def conversation_node(state: State) -> State:
    history = state.get("history", [])
    if not history:
        state["result"] = _base("conversation", "We haven't discussed anything yet in this chat.", True, "N/A")
        return state
    transcript = "\n\n".join(f"Q{i + 1}: {h['question']}\nA{i + 1}: {str(h.get('explanation', ''))[:600]}"
                             for i, h in enumerate(history))
    reply = llm.text(f"Here is our conversation so far:\n\n{transcript}\n\nAnswer this question about the "
                     f"conversation itself, using only what is written above: {state['question']}", max_tokens=500)
    state["result"] = _base("conversation", reply or "I ran into trouble with that, please try again.",
                            bool(reply), "N/A")
    return state


def clarify_node(state: State) -> State:
    parsed: ParsedQuestion = state["parsed"]
    q = parsed.clarifying_question or "Could you say which metric and period you mean?"
    state["result"] = _base("clarify", q, False, "N/A")
    return state


def off_topic_node(state: State) -> State:
    state["result"] = _base("off-topic", "I can only answer questions about Balaji Pharma's business and data.",
                            False, "N/A")
    return state


def _allowed_phrases(result: dict) -> list[str]:
    """Names that legitimately contain digits (products, periods), allowed in a recommendation."""
    phrases = dimension_values("product") + re.findall(r"FY\d{2}(?:-Q[1-4])?|\d{4}-Q[1-4]|Q[1-4]",
                                                       result.get("explanation", ""))
    return phrases


def finalize_node(state: State) -> State:
    result = state["result"]
    if result["found"] and result["route_decision"] in ("metric", "adhoc"):
        result["recommendation"] = llm.recommend(state["question"], result["explanation"],
                                                 result.get("facts", {}), _allowed_phrases(result))
    result.pop("facts", None)
    state["result"] = result
    return state


# ============================================================
# Wiring
# ============================================================

graph = StateGraph(State)
for name, fn in [("parse_node", parse_node), ("metric_node", metric_node), ("definition_node", definition_node),
                 ("adhoc_node", adhoc_node), ("conversation_node", conversation_node),
                 ("clarify_node", clarify_node), ("off_topic_node", off_topic_node),
                 ("finalize_node", finalize_node)]:
    graph.add_node(name, fn)

graph.add_edge(START, "parse_node")
graph.add_conditional_edges("parse_node", route, {n: n for n in [
    "metric_node", "definition_node", "adhoc_node", "conversation_node", "clarify_node", "off_topic_node"]})
for n in ["metric_node", "definition_node", "adhoc_node", "conversation_node", "clarify_node", "off_topic_node"]:
    graph.add_edge(n, "finalize_node")
graph.add_edge("finalize_node", END)

compiled_graph = graph.compile()


# ============================================================
# Public entry point
# ============================================================

def run_agent(question: str, history: list[dict] | None = None) -> dict:
    """
    Answer one question.

    history: earlier turns of THIS chat, oldest first. Each item:
             {"question": ..., "explanation": ..., "query": <previous structured query, optional>}
    """
    history = [{"question": h.get("question", ""), "explanation": h.get("explanation", ""),
                "query": h.get("query")} for h in (history or [])]
    if len(history) >= MAX_TURNS:
        return _base("limit-reached", LIMIT_MESSAGE, False, "N/A")

    start = time.time()
    state = compiled_graph.invoke({"question": question, "history": history})
    result = state["result"]
    print(f"DEBUG: run_agent took {time.time() - start:.2f}s route={result['route_decision']}")
    return result
