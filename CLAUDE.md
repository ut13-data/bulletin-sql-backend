# bUlleTin Backend — Dashboard Replication Rules

## Pattern for a new KPI dashboard
Every dashboard's data lives inside `main.py` as a self-contained block:
1. SQL query (or queries) computing the KPI, following the same style as 
   the Gross Margin section (CTEs, same join patterns, same rounding)
2. A response payload assembled under a new key, matching the naming 
   convention `xData` used for `grossMarginData` / `inventoryTurnoverData`
3. That key gets added to the `/all-data` endpoint response

When I ask for a new dashboard "X", replicate this exact structure for X's 
SQL and endpoint logic. Copy the Gross Margin block as the template unless 
told otherwise.

## Protected — do not modify without explicit permission
- Any existing key/query for Revenue, Gross Margin, or Inventory Turnover
- CORS config, DB connection setup, `/all-data` route signature (only ADD 
  a key to the response dict, never restructure it)

## Allowed
- Adding a new SQL block + new key to the `/all-data` response for the new KPI
- Adding new helper functions if the new KPI needs unique logic

## Process
1. Ask me for the KPI formula/business logic if I haven't given it
2. Show me the SQL you plan to write before adding it to main.py
3. Show me a diff-style summary of exactly what changes in main.py
4. Do not touch anything else in the file