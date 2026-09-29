# bUlleTin: Supply chain analytics dashboards for Balaji Pharma

KPI dashboards and an AI assistant for **Balaji Pharma**, a fictional Ayurvedic medicine manufacturer in Indore.
The company and its data are synthetic, modelled on 8 years of running a real pharma distribution business.

This is the **dashboard version**: a Next.js (React) front end on Vercel and a FastAPI (Python) back end on Render.
A chat-only version of the assistant is built in Streamlit (see "Related").

| Part | Repo | Hosted on |
|---|---|---|
| Front end (Next.js, React, Tailwind, Recharts) | `bulletin-balaji-pharma` | Vercel: https://bulletin-balaji-pharma.vercel.app/ |
| Back end (FastAPI, SQLite, pandas, LangGraph) | `bulletin-sql-backend` ||

---

## What it does

**Four dashboards**, each answering one business question:

| Dashboard | Business question | KPI cards | Main charts |
|---|---|---|---|
| Revenue | Are we growing, and where does revenue come from? | Net revenue (latest full fiscal year), YoY growth, discount %, revenue per customer | Monthly trend with discount %, YoY growth by month, share by distributor and category, top products |
| Gross Margin | Which products actually make money? | Gross margin %, gross profit | Quarterly margin trend vs benchmark, margin by category, revenue vs COGS by fiscal quarter |
| Inventory Turnover | How fast does stock move? | Annualised turnover, average inventory value | Turnover by category and by year-quarter, shelf life vs turnover, fast movers and overstock risks |
| DIO | How many days of stock do we hold? | Days inventory outstanding, annualised turnover | DIO by category, slowest-moving products |

Each dashboard also has status colours (Healthy / Watch / Attention / Critical), plain-language insights,
and an **Ask bUlleTin** tab. An overview page combines them into a health score and risk radar.

**Ask bUlleTin** answers questions like "Forecast revenue for the next 3 months", "Which category has the
lowest margin?" or "What if raw material costs rise 10%?", with the numbers, a chart, a recommendation, a
confidence level, and a Details section showing the formulas, the numbers table and the exact SQL that ran.

## The core idea

> The AI only interprets the question. Tested Python code calculates every number.

```
Question (React) -> /api/narrate (Next.js route) -> POST /agent-query (FastAPI)
  -> AI fills a fixed form (metric, period, filters, operation), validated by Pydantic
  -> one set of metric definitions calculates the numbers (SQL + pandas)
  -> code writes the answer text, chart, notes and confidence
  -> AI adds one recommendation sentence (code rejects it if it contains any number)
  -> JSON back to React (AIResponseCard)
```

The dashboards use the **same metric definitions** as the assistant, so a KPI card and a chat answer can
never disagree.

Why: in the first version the AI wrote its own SQL and reported numbers. Testing showed it read "FY24" as
calendar 2024, fitted a forecast on newest-first data (predicting a decline during real growth), and routed
comparison questions to a document search that cannot compare numbers. Moving all calculation into code fixed
this class of error rather than patching prompts one by one.

## Findings from building it

1. **Two different cost numbers.** The product table's standard cost is about 37% lower than actual production
   cost. The dashboard used it for category margins (52 to 55% everywhere) while the company-level margin was
   about 27%. COGS rebuilt from production batch costs matches the finance table every month (within 0.01%).
2. **A range selling below cost.** On actual production cost, Arishta has a gross margin of about -19%,
   and Classical and Syrup are around 4%. The Gross Margin dashboard now flags this as Critical.
3. **Inventory turnover was a 3-year total.** It showed 50.6, which is really about 17.4 per year.
   Now turnover x DIO = 365, as it should.
4. **Quarterly turnover mixed years.** "Q1" combined Q1 of 2022, 2023 and 2024. It now has one point per
   year-quarter (2022-Q1 to 2024-Q4).
5. **Seasonal sales need a seasonal model.** Tested on the last 6 known months, a straight-line forecast
   missed revenue by 12.3% on average; Holt-Winters missed by 2.2% and is chosen automatically.

## Architecture

```
Next.js on Vercel (front end)
  app/<dashboard>/page.tsx      server components, fetch data via lib/getDashboardData.ts
  app/<dashboard>/insights.ts   status colours and insight rules
  app/AIResponseCard.tsx        renders answers: explanation, chart, notes, evidence, recommendation, Details
  app/api/narrate/route.ts      forwards Ask questions to the back end (graph pipeline by default)
        |
        |  NEXT_PUBLIC_API_URL
        v
FastAPI on Render (back end)
  main.py                       dashboard endpoints + /agent-query
  agent/metrics.py              the metric catalog: every formula defined once, SQL built from it
  agent/periods.py              fiscal years (April to March), partial-period handling
  agent/operations.py           value / compare / trend / breakdown / forecast / what-if
  agent/forecasting.py          Holt-Winters vs seasonal naive vs straight line, chosen by backtest
  agent/parser.py               question -> validated form (Groq, gpt-oss-120b)
  agent/graph.py                LangGraph routing: metric | definition | ad-hoc | conversation
  agent/rag.py                  FAISS search over the 2 business documents
  agent/adhoc.py                fallback AI-written SQL, labelled Low confidence
  bulletin.db                   SQLite star schema (read-only)
```

## API

| Endpoint | Returns |
|---|---|
| `GET /revenue`, `/gross-margin`, `/inventory-turnover`, `/dio` | One dashboard's data, e.g. `{"revenue": {...}}` |
| `GET /all-data` | All four dashboards (used by the overview page) |
| `POST /agent-query` | `{"question", "thread_id"}` -> explanation, evidence, recommendation, confidence, chart, notes, table, definitions, sql |
| `POST /rag-query` | Answer from the business documents with sources |
| `GET /debug-thread/{thread_id}` | The conversation history kept for that thread |

## Metric definitions (examples)

| Metric | Definition |
|---|---|
| Net revenue | Sum of LineRevenue x (1 - DiscountPct): sales after discounts |
| COGS | Units sold x each product's average production cost (FactProductionBatches) |
| Gross margin | (Net revenue - COGS) / net revenue |
| Inventory turnover | Cost of units sold / average inventory value, annualised |
| DIO | 365 / annualised inventory turnover |
| Supplier on-time delivery | PO lines delivered on time / all PO lines |

All ratios are ratio-of-sums. Fiscal year runs April to March (FY24 = Apr 2023 to Mar 2024).
The full list is in `agent/metrics.py`.

## Key design decisions

| Decision | Why | Without it |
|---|---|---|
| One metric catalog shared by dashboards and assistant | Every formula defined once | A KPI card and a chat answer disagree |
| Unchanged JSON keys when the calculations were fixed | Front-end components kept working | Every chart component rewritten |
| Answer text written by code | Stated numbers always equal computed numbers | AI can round, swap or invent figures |
| Details section with the exact SQL | Anyone can check where a number came from | "Trust me" answers |
| Recommendation may not contain digits | Advice can't introduce unverified numbers | Advice contradicting the answer |
| Backend URL from an environment variable | Same code runs locally and in production | Hard-coded URLs to change by hand |
| `allow_origin_regex` for CORS | Works for Vercel production and preview URLs | Preview deployments blocked |

## Run it locally

**Back end** (`bulletin-sql-backend`)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload          # http://127.0.0.1:8000/docs
```
`.env` (never committed):
```
GROQ_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
```

**Front end** (`bulletin-balaji-pharma`)
```
npm install
npm run dev                        # http://localhost:3000
```
`.env.local` (never committed):
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
GROQ_API_KEY=...                   # used only by the older Ask modes in Settings
```
On Vercel, set `NEXT_PUBLIC_API_URL` to the Render URL.

## Testing

The shared `agent/` package is tested in the Streamlit repo (`python -m pytest -q tests`, 50 tests):
every metric recalculated independently in pandas, COGS and gross sales reconciled to the finance table
month by month, the SQL shown in Details re-run to confirm it gives the same number, and fake-AI tests for
messy output, unknown metrics, periods with no data and recommendations containing numbers.

## Limits

- Data covers Jan 2022 to Dec 2024; relative words like "last quarter" refer to the latest month with data.
- In this version, Ask bUlleTin keeps conversation history in the back end's memory, so it resets when
  Render restarts. (The Streamlit version stores chats in Supabase.)
- Forecasts assume past patterns continue; what-if scenarios are simple driver models.
- The Render free tier sleeps when idle, so the first request after a pause can take up to a minute.
- Ad-hoc questions outside the metric catalog rely on AI-written SQL and are marked Low confidence.

## Tech stack

**Front end:** Next.js, React, TypeScript, Tailwind CSS, Recharts, Vercel.
**Back end:** Python, FastAPI, SQLite, SQL, pandas, statsmodels, Pydantic, LangGraph, Groq (gpt-oss-120b),
FAISS + Hugging Face embeddings, Render.

## Related

- **bUlleTin chat (Streamlit):** the same assistant as a standalone chat app with login, saved chats and a
  daily question limit. Repo: `bulletin-streamlit` · Live: _add link_

Built by Utkarsh Pathak, with an AI coding assistant. Business rules, metric definitions and validation
of the results are my own.
