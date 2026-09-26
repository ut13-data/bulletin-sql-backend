"""
Turning "FY24", "Q3 2024", "last 6 months" into exact date ranges.

The LLM never does date math. It only names a period in a fixed format
(a PeriodSpec); this module converts it to [start, end) dates, clips it to the
data that actually exists, and says so when a period is only partly covered.

Balaji Pharma's fiscal year runs April to March: FY24 = 1 Apr 2023 to 31 Mar 2024.
"""
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from agent.db import read_sql

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_LOOKUP = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}
_MONTH_LOOKUP.update({
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "sept": 9, "october": 10, "november": 11, "december": 12,
})

PeriodType = Literal[
    "all", "fiscal_year", "calendar_year", "fiscal_quarter", "quarter", "month", "last_n_months", "range"
]


class PeriodError(ValueError):
    """Raised when a period can't be understood or has no data. The message is shown to the user."""


class PeriodSpec(BaseModel):
    """What the LLM produces. Only one of value / n / start+end is used, depending on type."""
    type: PeriodType = "all"
    value: str | int | None = None   # "FY24", 2024, "2024-Q3", "FY25-Q2", "2024-11", "latest", "previous"
    n: int | None = None             # for last_n_months
    start: str | None = None         # for range: "2023-01-01" or "2023-01"
    end: str | None = None           # for range, inclusive: "2023-06-30" or "2023-06"


@dataclass(frozen=True)
class ResolvedPeriod:
    label: str             # "FY24 (Apr 2023 to Mar 2024)"
    short: str             # "FY24"
    start: date            # inclusive, clipped to available data
    end: date              # exclusive, clipped to available data
    requested_start: date
    requested_end: date
    kind: str

    @property
    def start_str(self) -> str:
        return self.start.isoformat()

    @property
    def end_str(self) -> str:
        return self.end.isoformat()

    @property
    def months(self) -> int:
        return months_between(self.start, self.end)

    @property
    def requested_months(self) -> int:
        return months_between(self.requested_start, self.requested_end)

    @property
    def is_partial(self) -> bool:
        return (self.start, self.end) != (self.requested_start, self.requested_end)

    @property
    def coverage_note(self) -> str:
        if not self.is_partial:
            return ""
        return (f"{self.short} has data for {month_label(self.start)} to "
                f"{month_label(add_months(self.end, -1))} only "
                f"({self.months} of {self.requested_months} months).")


# ------------------------------------------------------------
# Small date helpers (month granularity is all we need)
# ------------------------------------------------------------

def add_months(d: date, n: int) -> date:
    total = d.year * 12 + (d.month - 1) + n
    return date(total // 12, total % 12 + 1, 1)


def months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def month_label(d: date) -> str:
    return f"{MONTH_NAMES[d.month - 1]} {d.year}"


def fiscal_year_of(d: date) -> int:
    """Fiscal year END year: Apr 2023 to Mar 2024 -> 2024 (FY24)."""
    return d.year + 1 if d.month >= 4 else d.year


def fy_label(end_year: int) -> str:
    return f"FY{end_year % 100:02d}"


# ------------------------------------------------------------
# The data window: every answer is limited to months that have sales data
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def data_window() -> tuple[date, date]:
    """[first month, month after last month] of FactSalesLines. Cached: data is read-only."""
    row = read_sql("SELECT MIN(substr(OrderDate, 1, 10)) AS lo, MAX(substr(OrderDate, 1, 10)) AS hi FROM FactSalesLines").iloc[0]
    lo = date.fromisoformat(row["lo"])
    hi = date.fromisoformat(row["hi"])
    return date(lo.year, lo.month, 1), add_months(date(hi.year, hi.month, 1), 1)


def data_window_text() -> str:
    lo, hi = data_window()
    return f"{month_label(lo)} to {month_label(add_months(hi, -1))}"


def available_periods_text() -> str:
    """Plain-text list of periods with data, given to the LLM so it names real periods."""
    lo, hi = data_window()
    last = add_months(hi, -1)
    fys = []
    for fy in range(fiscal_year_of(lo), fiscal_year_of(last) + 1):
        p = resolve(PeriodSpec(type="fiscal_year", value=fy_label(fy)))
        fys.append(f"{p.short} ({month_label(p.start)} to {month_label(add_months(p.end, -1))}"
                   + (", partial" if p.is_partial else "") + ")")
    years = []
    for y in range(lo.year, last.year + 1):
        p = resolve(PeriodSpec(type="calendar_year", value=y))
        years.append(f"{y}" + (" (partial)" if p.is_partial else ""))
    return (f"Data covers {data_window_text()}. Latest month with data: {last.strftime('%Y-%m')}.\n"
            f"Fiscal years: {', '.join(fys)}.\nCalendar years: {', '.join(years)}.")


# ------------------------------------------------------------
# Parsing individual period formats
# ------------------------------------------------------------

def _parse_fiscal_year(value) -> int:
    """'FY24', 'FY2024', '2023-24', '2023-2024', 24, 2024 -> 2024 (the year the FY ends)."""
    s = str(value).strip().upper().replace(" ", "")
    m = re.fullmatch(r"(?:FY)?(\d{4})[-/](\d{2}|\d{4})", s)
    if m:
        return int(m.group(1)) + 1
    m = re.fullmatch(r"(?:FY)?(\d{2}|\d{4})", s)
    if m:
        y = int(m.group(1))
        return 2000 + y if y < 100 else y
    raise PeriodError(f"I couldn't read the fiscal year '{value}'. Try a format like FY24.")


def _parse_month(value) -> date:
    """'2024-11', '2024-11-01', 'Nov 2024', 'November 2024' -> date(2024, 11, 1)."""
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?(?:[ T].*)?", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    m = re.fullmatch(r"([A-Za-z]+)[\s,-]+(\d{4})", s)
    if m and m.group(1).lower() in _MONTH_LOOKUP:
        return date(int(m.group(2)), _MONTH_LOOKUP[m.group(1).lower()], 1)
    raise PeriodError(f"I couldn't read the month '{value}'. Try a format like 2024-11.")


def _parse_day(value, *, end: bool) -> date:
    """Range endpoints: a full date, or a month (start of month / end of month)."""
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?", s)
    if m:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return date.fromordinal(d.toordinal() + 1) if end else d
    first = _parse_month(s)
    return add_months(first, 1) if end else first


def _parse_quarter(value) -> tuple[int, int]:
    """'2024-Q3', 'Q3 2024', '2024Q3' -> (2024, 3)."""
    s = str(value).strip().upper().replace(" ", "")
    m = re.fullmatch(r"(\d{4})-?Q([1-4])", s) or re.fullmatch(r"Q([1-4])-?(\d{4})", s)
    if not m:
        raise PeriodError(f"I couldn't read the quarter '{value}'. Try a format like 2024-Q3.")
    a, b = m.groups()
    return (int(a), int(b)) if len(a) == 4 else (int(b), int(a))


def _parse_fiscal_quarter(value) -> tuple[int, int]:
    """'FY25-Q2', 'Q2 FY25', 'FY2025Q2' -> (2025, 2)."""
    s = str(value).strip().upper().replace(" ", "")
    m = re.fullmatch(r"FY(\d{2}|\d{4})-?Q([1-4])", s) or re.fullmatch(r"Q([1-4])-?FY(\d{2}|\d{4})", s)
    if not m:
        raise PeriodError(f"I couldn't read the fiscal quarter '{value}'. Try a format like FY25-Q2.")
    a, b = m.groups()
    fy, q = (a, b) if s.startswith("FY") else (b, a)
    fy = int(fy)
    return (2000 + fy if fy < 100 else fy), int(q)


RELATIVE_VALUES = {"latest", "current", "this", "previous", "last", "prior", "latest_complete"}


def _is_relative(value) -> bool:
    return str(value).strip().lower() in RELATIVE_VALUES


def _relative_offset(value) -> int:
    return 0 if str(value).strip().lower() in {"latest", "current", "this", "latest_complete"} else -1


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def resolve(spec: PeriodSpec | dict | None) -> ResolvedPeriod:
    """PeriodSpec -> exact [start, end) dates, clipped to the data window."""
    if spec is None:
        spec = PeriodSpec()
    if isinstance(spec, dict):
        spec = PeriodSpec.model_validate(spec)

    # "latest_complete": the most recent period of this type that has data for every month.
    if str(spec.value).strip().lower() == "latest_complete":
        latest = _resolve(spec.model_copy(update={"value": "latest"}))
        if not latest.is_partial:
            return latest
        return _resolve(spec.model_copy(update={"value": "previous"}))
    return _resolve(spec)


def _resolve(spec: PeriodSpec) -> ResolvedPeriod:
    lo, hi = data_window()
    last_month = add_months(hi, -1)
    t, v = spec.type, spec.value

    if t == "all":
        start, end, short = lo, hi, "All data"
        label = f"all available data ({data_window_text()})"

    elif t == "fiscal_year":
        if v is None or _is_relative(v):
            fy = fiscal_year_of(last_month) + _relative_offset(v or "latest")
        else:
            fy = _parse_fiscal_year(v)
        start, end = date(fy - 1, 4, 1), date(fy, 4, 1)
        short = fy_label(fy)
        label = f"{short} (Apr {fy - 1} to Mar {fy})"

    elif t == "calendar_year":
        if v is None or _is_relative(v):
            y = last_month.year + _relative_offset(v or "latest")
        else:
            m = re.search(r"\d{4}", str(v))
            if not m:
                raise PeriodError(f"I couldn't read the year '{v}'.")
            y = int(m.group())
        start, end = date(y, 1, 1), date(y + 1, 1, 1)
        short = str(y)
        label = f"calendar year {y}"

    elif t == "quarter":
        if v is None or _is_relative(v):
            q_start = date(last_month.year, 3 * ((last_month.month - 1) // 3) + 1, 1)
            start = add_months(q_start, 3 * _relative_offset(v or "latest"))
        else:
            y, q = _parse_quarter(v)
            start = date(y, 3 * (q - 1) + 1, 1)
        end = add_months(start, 3)
        short = f"{start.year}-Q{(start.month - 1) // 3 + 1}"
        label = f"{short} ({month_label(start)} to {month_label(add_months(end, -1))})"

    elif t == "fiscal_quarter":
        if v is None or _is_relative(v):
            months_into_fy = (last_month.month - 4) % 12
            q_start = add_months(last_month, -(months_into_fy % 3))
            start = add_months(q_start, 3 * _relative_offset(v or "latest"))
        else:
            fy, q = _parse_fiscal_quarter(v)
            start = add_months(date(fy - 1, 4, 1), 3 * (q - 1))
        end = add_months(start, 3)
        fq = ((start.month - 4) % 12) // 3 + 1
        short = f"{fy_label(fiscal_year_of(start))}-Q{fq}"
        label = f"{short} ({month_label(start)} to {month_label(add_months(end, -1))})"

    elif t == "month":
        if v is None or _is_relative(v):
            start = add_months(last_month, _relative_offset(v or "latest"))
        else:
            start = _parse_month(v)
        end = add_months(start, 1)
        short = label = month_label(start)

    elif t == "last_n_months":
        n = spec.n or (int(v) if str(v or "").isdigit() else None)
        if not n or n < 1:
            raise PeriodError("Please say how many months, e.g. 'last 6 months'.")
        end = hi
        start = add_months(end, -n)
        short = f"last {n} months"
        label = f"the last {n} months ({month_label(start)} to {month_label(last_month)})"

    elif t == "range":
        if not spec.start or not spec.end:
            raise PeriodError("A date range needs both a start and an end.")
        start = _parse_day(spec.start, end=False)
        end = _parse_day(spec.end, end=True)
        if end <= start:
            raise PeriodError("The end of the date range is before its start.")
        short = label = f"{start.isoformat()} to {date.fromordinal(end.toordinal() - 1).isoformat()}"

    else:  # pragma: no cover (pydantic already restricts type)
        raise PeriodError(f"Unknown period type '{t}'.")

    clipped_start, clipped_end = max(start, lo), min(end, hi)
    if clipped_start >= clipped_end:
        raise PeriodError(f"There is no data for {short}. Data covers {data_window_text()}.")
    if (clipped_start, clipped_end) != (start, end) and t != "all":
        label = (f"{short} (data for {month_label(clipped_start)} to "
                 f"{month_label(add_months(clipped_end, -1))} only)")

    return ResolvedPeriod(label=label, short=short, start=clipped_start, end=clipped_end,
                          requested_start=start, requested_end=end, kind=t)


def like_for_like(partial: ResolvedPeriod, other: ResolvedPeriod) -> ResolvedPeriod | None:
    """
    Same calendar months as `partial`, shifted into `other`.
    FY25 (data Apr-Dec 2024) vs FY24 -> FY24 restricted to Apr-Dec 2023.
    Returns None when the shift doesn't line up cleanly.
    """
    shift = months_between(partial.requested_start, other.requested_start)
    if shift % 12 != 0:
        return None
    start = add_months(partial.start, shift)
    end = add_months(partial.end, shift)
    lo, hi = data_window()
    if start < lo or end > hi:
        return None
    short = f"{other.short} ({month_label(start)} to {month_label(add_months(end, -1))})"
    return ResolvedPeriod(label=short, short=short, start=start, end=end,
                          requested_start=start, requested_end=end, kind="range")
