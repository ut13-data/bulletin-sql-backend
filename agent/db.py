"""
Read-only access to bulletin.db.

Every query in the app goes through here, so there is exactly one place that
guarantees: read-only connection, parameterised values, and a time limit.
"""
import sqlite3
import time
from functools import lru_cache

import pandas as pd

from agent.config import DB_PATH

QUERY_TIME_LIMIT_SECONDS = 10


def connect_read_only() -> sqlite3.Connection:
    # mode=ro: SQLite itself refuses any write, whatever SQL is sent.
    conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True, check_same_thread=False)

    # Abort runaway queries (e.g. an accidental cross join from ad-hoc SQL).
    deadline = time.monotonic() + QUERY_TIME_LIMIT_SECONDS
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    return conn


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Values always go in `params`, never into the SQL text."""
    conn = connect_read_only()
    try:
        return pd.read_sql_query(sql, conn, params=params or {})
    finally:
        conn.close()


@lru_cache(maxsize=1)
def table_columns() -> dict[str, list[str]]:
    """{table: [columns]} for every table. Cached: the database is read-only."""
    conn = connect_read_only()
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: [r[1] for r in conn.execute(f"PRAGMA table_info([{t}])")] for t in tables}
    finally:
        conn.close()
