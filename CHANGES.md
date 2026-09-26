# Backend changes: accuracy fixes

## What to do in the backend repo
1. Copy the `agent/` folder into the repo root (next to main.py). It is the same package the Streamlit app uses.
2. Replace `main.py` with the new one.
3. Delete `agent_graph.py` (replaced by `agent/graph.py`).
4. Add `statsmodels` and `langgraph` to requirements.txt (scikit-learn is no longer needed).

The JSON keys are unchanged, so frontend pages keep working.

## Numbers that change (current data)
| KPI | Old | New | Why |
|---|---|---|---|
| Total revenue (FY24) | Rs 20.71M | Rs 20.71M | unchanged |
| Discount % | 0.93% (all 3 years) | 0.97% (FY24) | same period as the revenue card next to it |
| Avg revenue / customer | Rs 314.89K (3 years) | Rs 113.18K (FY24) | same period as the revenue card |
| Gross margin | 26.58% | 25.89% | now after discounts (was on revenue before discounts) |
| Gross profit | Rs 16.14M | Rs 15.57M | same reason |
| Category margins | 52% to 55% for every category | -19% (Arishta) to 61% (Capsule) | old used DimProduct.UnitCost, about 37% below actual production cost; categories now reconcile with the company margin |
| Inventory turnover | 50.61 | 17.40 | old was a 3-year total shown as if yearly; now annualised, so turnover x DIO = 365 |
| Average inventory value | Rs 5.53L | Rs 8.48L | valued at production cost, same basis as COGS |
| DIO | 21.7 days | 21.0 days | same cost basis; counts the days in every week of data |
| Quarterly turnover | 4 points (Q1 of all years merged) | 12 points (2022-Q1 ... 2024-Q4) | the old chart mixed different years; `quarter` is now a label like "2024-Q3" |

## Bugs fixed in the agent (agent_graph.py -> agent/graph.py)
- "FY24" read as calendar 2024
- forecast fitted on newest-first data (reversed trend)
- "most recent months" query returned the oldest months
- comparison questions routed to the document search, which can't compare numbers
- straight-line forecast on strongly seasonal sales (backtest error 12.3%); now Holt-Winters is chosen by backtest (2.2%)
- recommendation could contradict the answer; now written from the computed facts and may not state numbers
- META_EXAMPLES and the similarity threshold were defined twice (0.3 vs 0.75); meta questions are now detected by the parser
