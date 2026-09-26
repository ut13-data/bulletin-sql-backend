"""
All calls to the language model go through here.

The LLM is used for LANGUAGE only:
  * understanding the question (parser.py)
  * writing one recommendation sentence (checked in code, see recommend())
  * ad-hoc SQL for questions outside the metric catalog (adhoc.py, clearly labelled)
  * answering from documents and summarising the conversation
It never produces the numbers stated in a catalog answer.
"""
import json
import re
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from agent.config import MODEL, get_secret


@lru_cache(maxsize=1)
def get_client():
    from groq import Groq
    return Groq(api_key=get_secret("GROQ_API_KEY"))


def chat(messages: list[dict], **kwargs) -> dict:
    """One Groq call. Never raises: returns {"ok": False, "error": ...} on failure."""
    kwargs.setdefault("model", MODEL)
    try:
        response = get_client().chat.completions.create(messages=messages, **kwargs)
        return {"ok": True, "message": response.choices[0].message}
    except Exception as e:
        print(f"DEBUG: LLM call failed: {e}")
        return {"ok": False, "error": str(e)}


def structured(messages: list[dict], schema: type[BaseModel], **kwargs) -> tuple[BaseModel | None, str | None]:
    """
    JSON-mode call validated against a pydantic schema. If the first reply is
    invalid, the validation error is sent back once so the model can fix it.
    """
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("max_tokens", 1500)
    kwargs.setdefault("reasoning_effort", "low")
    error = None
    for attempt in range(2):
        result = chat(messages, response_format={"type": "json_object"}, **kwargs)
        if not result["ok"]:
            return None, result["error"]
        content = result["message"].content or ""
        try:
            return schema.model_validate(json.loads(content)), None
        except (json.JSONDecodeError, ValidationError) as e:
            error = str(e)[:800]
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That JSON was invalid: {error}\nReturn ONLY corrected JSON."},
            ]
    return None, error


def text(prompt: str, **kwargs) -> str | None:
    kwargs.setdefault("temperature", 0.2)
    kwargs.setdefault("max_tokens", 400)
    kwargs.setdefault("reasoning_effort", "low")
    result = chat([{"role": "user", "content": prompt}], **kwargs)
    return (result["message"].content or "").strip() if result["ok"] else None


# ============================================================
# Recommendation: one sentence, and it may not state numbers
# ============================================================

def _contains_unverified_digits(sentence: str, allowed_phrases: list[str]) -> bool:
    """True if the sentence contains digits that aren't part of a known name (e.g. 'FY24', '500g')."""
    scrubbed = sentence
    for phrase in sorted(allowed_phrases, key=len, reverse=True):
        if phrase:
            scrubbed = re.sub(re.escape(phrase), " ", scrubbed, flags=re.IGNORECASE)
    return bool(re.search(r"\d", scrubbed))


class _Rec(BaseModel):
    recommendation: str = ""


def recommend(question: str, explanation: str, facts: dict, allowed_phrases: list[str]) -> str:
    """
    One actionable sentence based on the computed answer. The model is told not
    to write numbers, and code enforces it: if a digit appears (outside product
    or period names), the sentence is rejected. That way a recommendation can't
    introduce a figure that was never calculated.
    """
    prompt = (
        "You advise the management of Balaji Pharma, an Ayurvedic medicine manufacturer.\n"
        "Below is a verified answer to a business question. Write ONE short, concrete, actionable "
        "recommendation (max 25 words) that follows from it.\n"
        "RULES: do NOT write any numbers, percentages, amounts or dates; refer to things by name only "
        "(e.g. 'the Arishta range', 'the winter peak'). Do not restate the answer. If no action is "
        'warranted, return an empty string.\nRespond ONLY with JSON: {"recommendation": "..."}\n\n'
        f"Question: {question}\nVerified answer: {explanation}\nKey facts: {json.dumps(facts, default=str)[:2000]}"
    )
    for _ in range(2):
        rec, _err = structured([{"role": "user", "content": prompt}], _Rec, max_tokens=300, temperature=0.2)
        if rec is None:
            return ""
        sentence = rec.recommendation.strip()
        if not sentence:
            return ""
        if not _contains_unverified_digits(sentence, allowed_phrases):
            return sentence
        prompt += "\n\nYour previous answer contained numbers. Rewrite it with NO digits at all."
    return ""
