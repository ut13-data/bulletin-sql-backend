"""
Monthly forecasting with an honest accuracy check.

Instead of trusting one model, we:
  1. Hold back the most recent months, forecast them from the older data,
     and measure the error (a backtest).
  2. Pick the method with the lowest backtest error.
  3. Refit that method on all the data and forecast forward.
  4. Report the backtest error (MAPE) and an error-based likely range,
     so the user sees how far off this method has actually been.

Methods compared:
  * Holt-Winters: exponential smoothing with trend + 12-month seasonality.
  * Seasonal naive + growth: same month last year x recent year-on-year growth.
  * Linear trend: straight line (the old approach, kept as a baseline).
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_MONTHS_SEASONAL = 24   # need two full years to learn a 12-month pattern
BACKTEST_MONTHS = 6
MAX_HORIZON = 12


@dataclass
class ForecastResult:
    method: str
    method_label: str
    history: pd.Series            # index: 'YYYY-MM', values
    forecast: pd.Series           # index: 'YYYY-MM' (future months)
    low: pd.Series
    high: pd.Series
    backtest_mape: float | None   # average % miss on held-back months
    candidates: dict              # method -> backtest MAPE (for the Details section)
    notes: list


METHOD_LABELS = {
    "holt_winters": "Holt-Winters (trend + 12-month seasonality)",
    "seasonal_naive_growth": "Same month last year x year-on-year growth",
    "linear_trend": "Straight-line trend",
}


def _next_months(last: str, n: int) -> list[str]:
    p = pd.Period(last, freq="M")
    return [str(p + i) for i in range(1, n + 1)]


def _fit_predict(method: str, y: np.ndarray, horizon: int) -> np.ndarray:
    if method == "linear_trend":
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * np.arange(len(y), len(y) + horizon)

    if method == "seasonal_naive_growth":
        # Trailing-12 vs previous-12 growth, applied to the same month last year.
        growth = y[-12:].sum() / y[-24:-12].sum() if len(y) >= 24 and y[-24:-12].sum() > 0 else 1.0
        return np.array([y[len(y) - 12 + (h % 12)] * growth ** (1 + h // 12) for h in range(horizon)])

    if method == "holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "mul" if (y > 0).all() else "add"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seasonal,
                                         seasonal_periods=12, initialization_method="estimated").fit()
        return np.asarray(model.forecast(horizon))

    raise ValueError(method)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def forecast_monthly(history: pd.Series, horizon: int = 3) -> ForecastResult:
    """history: monthly values indexed by 'YYYY-MM', oldest first, no gaps."""
    history = history.dropna().astype(float).sort_index()
    horizon = max(1, min(int(horizon), MAX_HORIZON))
    y = history.to_numpy()
    n = len(y)
    notes = []

    if n < 6:
        raise ValueError("At least 6 months of history are needed to forecast.")

    methods = ["linear_trend"]
    if n >= MIN_MONTHS_SEASONAL + BACKTEST_MONTHS:
        methods = ["holt_winters", "seasonal_naive_growth", "linear_trend"]
    elif n >= MIN_MONTHS_SEASONAL:
        methods = ["seasonal_naive_growth", "linear_trend"]
    else:
        notes.append(f"Only {n} months of history, so seasonality can't be modelled; using a straight-line trend.")

    # 1. Backtest: forecast the last BACKTEST_MONTHS from the data before them.
    bt = min(BACKTEST_MONTHS, n // 4)
    candidates = {}
    pct_errors = {}
    for m in methods:
        train = y[:-bt]
        if m == "seasonal_naive_growth" and len(train) < 24:
            continue
        if m == "holt_winters" and len(train) < MIN_MONTHS_SEASONAL:
            continue
        try:
            pred = _fit_predict(m, train, bt)
            candidates[m] = round(_mape(y[-bt:], pred), 2)
            pct_errors[m] = (y[-bt:] - pred) / y[-bt:]
        except Exception as e:  # a method failing must not break the answer
            notes.append(f"{METHOD_LABELS[m]} could not be fitted ({type(e).__name__}).")

    best = min(candidates, key=candidates.get) if candidates else "linear_trend"

    # 2. Refit the winner on all data and forecast.
    pred = _fit_predict(best, y, horizon)
    future_idx = _next_months(history.index[-1], horizon)
    forecast = pd.Series(pred, index=future_idx)

    # 3. Likely range from the winner's own backtest misses (80th percentile of % error).
    if best in pct_errors and len(pct_errors[best]):
        band = float(np.quantile(np.abs(pct_errors[best]), 0.8))
    else:
        band = 0.15
        notes.append("No backtest was possible, so the likely range is a rough +/-15%.")
    band = max(band, 0.02)
    # The range widens with distance: +10% per extra month ahead.
    widen = np.array([1 + 0.1 * h for h in range(horizon)])
    low = forecast * (1 - band * widen)
    high = forecast * (1 + band * widen)

    return ForecastResult(
        method=best, method_label=METHOD_LABELS[best], history=history, forecast=forecast,
        low=low, high=high, backtest_mape=candidates.get(best), candidates=candidates, notes=notes,
    )


def confidence_from_mape(mape: float | None, n_months: int) -> str:
    if mape is None or n_months < MIN_MONTHS_SEASONAL:
        return "Low"
    if mape <= 5:
        return "High"
    if mape <= 12:
        return "Moderate"
    return "Low"
