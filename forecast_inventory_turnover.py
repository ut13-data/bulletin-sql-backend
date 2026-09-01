import pandas as pd
import sqlite3
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

conn = sqlite3.connect("bulletin.db")

base = pd.read_sql("""
    SELECT
        f.ProductID,
        p.ShelfLifeMonths,
        SUM(f.Sold) AS units_sold,
        AVG((f.OpeningStock + f.ClosingStock) / 2.0) AS avg_inventory_units
    FROM FactInventoryWeekly f
    JOIN DimProduct p ON f.ProductID = p.ProductID
    GROUP BY f.ProductID
""", conn)

conn.close()

base["inventory_turnover"] = base["units_sold"] / base["avg_inventory_units"]

X = base[["ShelfLifeMonths"]]
y = base["inventory_turnover"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
r2 = r2_score(y, predictions)

print(f"R2 score: {r2:.3f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Shelf Life coefficient: {model.coef_[0]:.4f}")

output = {
    "intercept": model.intercept_,
    "coefShelfLifeMonths": model.coef_[0],
    "r2Score": r2,
    "note": "Shelf life explains only ~6% of turnover variation (R2 = 0.06). This is a weak relationship — turnover is likely driven primarily by demand, not shelf life. Treat this prediction as low-confidence."
}

json_path = "../../balaji-pharma-intelligence/app/inventory-turnover/forecast_model.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nforecast_model.json written.")