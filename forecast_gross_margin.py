import pandas as pd
import sqlite3
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

conn = sqlite3.connect("bulletin.db")

quarterly = pd.read_sql("""
    SELECT
        CAST(strftime('%Y', MonthYear) AS INTEGER) AS year,
        (CAST(strftime('%m', MonthYear) AS INTEGER) - 1) / 3 + 1 AS quarter,
        SUM(GrossProfit) AS gross_profit,
        AVG(GrossMarginPct) * 100 AS gross_margin_pct
    FROM FactFinanceMonthly
    GROUP BY year, quarter
    ORDER BY year, quarter
""", conn)

quarterly["quarter_index"] = range(1, len(quarterly) + 1)

X = quarterly[["quarter_index"]]
y = quarterly["gross_margin_pct"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
r2 = r2_score(y, predictions)

print(f"\nR2 score: {r2:.3f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Quarter index coefficient: {model.coef_[0]:.2f}")

conn.close()

print(quarterly)

output = {
    "intercept": model.intercept_,
    "coefQuarterIndex": model.coef_[0],
    "r2Score": r2,
    "lastQuarterIndex": int(quarterly["quarter_index"].max()),
    "note": "Linear regression on gross margin % vs. quarter index (12 quarters). R2 is moderate — quarter alone explains less than half of margin variation; other factors (category mix, COGS shifts) likely matter more."
}

json_path = "../../balaji-pharma-intelligence/app/gross-margin/forecast_model.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nforecast_model.json written.")