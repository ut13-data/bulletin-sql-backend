"""Number formatting in one place (Indian lakh/crore style for rupees)."""
import math

from agent.metrics import metric


def indian_group(n: float, decimals: int = 0) -> str:
    """1234567.8 -> '12,34,568' (Indian digit grouping)."""
    neg = n < 0
    n = abs(n)
    s = f"{n:.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = whole + (f".{frac}" if frac else "")
    return f"-{out}" if neg else out


def inr(v: float, exact: bool = False) -> str:
    """Rupees. Short form: ₹2.07 Cr, ₹45.30 L, ₹12,345. exact=True: full digits."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if exact:
        return f"₹{indian_group(v, 0)}"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} L"
    return f"{sign}₹{indian_group(a, 0)}"


def fmt(metric_key: str, v: float, exact: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    unit = metric(metric_key).unit
    if unit == "inr":
        return inr(v, exact=exact)
    if unit == "pct":
        return f"{v:.1f}%"
    if unit == "ratio":
        return f"{v:.1f}x"
    if unit == "days":
        return f"{v:.1f} days"
    return indian_group(v, 0)


def fmt_change(metric_key: str, old: float, new: float) -> str:
    """'+12.4%' for amounts and counts; '+1.3 pts' for percentages (a change in a rate)."""
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in (old, new)):
        return "n/a"
    unit = metric(metric_key).unit
    if unit == "pct":
        return f"{new - old:+.1f} pts"
    if old == 0:
        return "n/a"
    return f"{(new - old) / abs(old) * 100:+.1f}%"
