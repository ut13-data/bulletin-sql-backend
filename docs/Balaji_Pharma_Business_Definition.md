# Balaji Pharma — Business Definition Document
### Ayurvedic Pharmaceutical Manufacturing SME | Indore, India
*Foundational document for Supply Chain Analytics Portfolio Project*

---

## 1. Business Overview

**Company Name:** Balaji Pharma
**Industry:** Ayurvedic Pharmaceutical Manufacturing (FMCG-Pharma hybrid)
**Founded:** August 2018
**Location:** Indore, Madhya Pradesh, India (single manufacturing unit + central warehouse)
**Ownership:** Founder-led, owner-operated SME (no external investors)
**Scale:** ~15 active SKUs, ~150 retail outlets, 7–8 raw material/packaging suppliers, single production facility

**Positioning:** Balaji Pharma manufactures traditional Ayurvedic formulations (churna, tablets, syrups, oils, capsules) targeting immunity, digestion, respiratory health, and general wellness. It operates as a regional challenger brand in Madhya Pradesh, with early-stage expansion into e-commerce and B2B marketplaces.

**Business Model:** Vertically integrated — the company procures raw herbs and packaging, manufactures in-house under batch production, and distributes through a hybrid of traditional trade (distributors + retail outlets) and modern/digital channels (Amazon, IndiaMART, OTC counter sales).

**Revenue Character:**
- Primary revenue: Distributor + Retail trade (bulk of volume, lower margin, high volume)
- Secondary: OTC direct sales (in and around Indore, cash-and-carry style)
- Emerging: Amazon (D2C, higher margin, small volume, growing fastest)
- B2B lead generation: IndiaMART (bulk/institutional inquiries, lumpy order pattern)

**Why this matters for analytics:** Every downstream dataset (sales, inventory, procurement) must reflect this — retail/distributor channel should carry ~70–75% of volume, OTC ~10–15%, Amazon ~8–12% (growing YoY), IndiaMART ~5–8% (irregular, large-ticket).

---

## 2. Organization Structure

Balaji Pharma runs lean by design — a real reflection of a founder-operated SME, not a large enterprise hierarchy.

```
                        Founder & General Manager
                         (Operations, P&L, Compliance)
                                    |
        --------------------------------------------------------
        |                 |                  |                 |
   Production        Procurement        Sales & Distribution   Quality/
   Supervisor        Executive          (Market Representatives) Compliance
   (1)               (1)                (6 Market Reps)         (Part-time/
        |                 |                  |                  Outsourced
   Production Staff   Vendor Coordination  Distributor Liaison  Consultant)
   (3–4 workers)                            (2 Distributors)
```

**Headcount reality (per founder's own record):**
- Total core team: 13 people (5 direct employees + 6 market representatives + 2 distributor partners)
- No dedicated IT/analytics function today — this is the gap this project fills
- Compliance and quality testing are typically outsourced/consultant-based in Ayurvedic SMEs of this size (Ayush license holders rarely carry a full-time QA department at 15-SKU scale)

**Reporting lines:**
- All departments report directly to the Founder & GM — flat structure, single point of decision-making
- Market Representatives report jointly to Sales function and interface with Distributors

---

## 3. Supply Chain Flow

```
RAW MATERIAL SUPPLIERS (7-8)          PACKAGING SUPPLIERS
  (Herbs, oils, excipients)             (Bottles, cartons, labels)
            \                                  /
             \                                /
              v                              v
                    PROCUREMENT & INBOUND STORE
                     (Raw Material Warehouse)
                                |
                                v
                     BATCH MANUFACTURING
              (Production Planning -> Batch Execution
                   -> In-process QC -> Finished Batch)
                                |
                                v
                     FINISHED GOODS WAREHOUSE
                        (Indore Central)
                                |
        --------------------------------------------------
        |               |                |                |
        v               v                v                v
   DISTRIBUTORS      RETAIL OUTLETS    AMAZON (D2C)    OTC COUNTER
   (2 distributors)  (~150 outlets     Fulfillment      SALES
                       via 6 reps)                      (Walk-in/local)
        |               |
        v               v
   Sub-distributors   End Consumer
   / Regional retail
                                
                                                    IndiaMART
                                              (B2B bulk inquiry -> 
                                               negotiated order -> 
                                               direct dispatch)
```

**Key supply chain characteristics:**
- Single-echelon manufacturing (no contract manufacturing / third-party production)
- Two-tier physical distribution: Distributor → Retail, and direct-to-retail via market reps
- Digital channels (Amazon, IndiaMART) ship directly from the Finished Goods Warehouse, bypassing distributors
- Raw material sourcing is the primary lead-time and quality risk point (herbs are seasonal, packaging is not)

---

## 4. Departments

| Department | Core Responsibility | Real-World SME Constraint |
|---|---|---|
| **Procurement** | Vendor management, PO issuance, raw material & packaging sourcing | Single procurement executive manages 7–8 vendor relationships manually (Excel/WhatsApp-driven, no formal ERP negotiation workflow) |
| **Production/Manufacturing** | Batch planning, mixing/formulation, filling, packing | Capacity-constrained by machinery (1 production line), batch-size fixed by kettle/mixer capacity |
| **Quality & Compliance** | Raw material testing, in-process QC, AYUSH regulatory compliance, batch release | Likely outsourced/consultant model at this scale; compliance tied to state Ayush licensing |
| **Inventory & Warehousing** | Raw material store, finished goods store, stock reconciliation | Manual/Excel-based stock registers typical; no WMS |
| **Sales & Distribution** | Retail servicing, distributor coordination, market visits | 6 market reps covering ~150 outlets (~25 outlets/rep) — realistic FMCG rep-to-outlet ratio |
| **Finance & Accounts** | Invoicing, payments, P&L, statutory compliance (GST) | Likely handled by founder + part-time accountant/CA, not a full finance team |
| **E-commerce/Digital** | Amazon listing management, IndiaMART lead handling | Newest function, likely managed part-time by founder or a junior hire post-2020 |

---

## 5. Business Processes

**5.1 Procure-to-Pay (P2P)**
Demand forecast → Purchase requisition → Vendor selection/PO → Goods receipt & QC → Vendor invoice → Payment

**5.2 Plan-to-Produce (Manufacturing)**
Sales forecast + safety stock → Production plan (weekly/monthly) → Raw material issue → Batch manufacturing → In-process QC → Batch release → Finished goods put-away

**5.3 Order-to-Cash (O2C)**
Order received (Distributor/Retail/Amazon/IndiaMART/OTC) → Order validation & stock check → Dispatch → Invoice → Payment collection → Reconciliation

**5.4 Inventory Management**
Continuous review for fast-moving SKUs, periodic review for slow-movers; FEFO (First-Expiry-First-Out) issuance due to shelf-life constraints on Ayurvedic formulations

**5.5 Demand Planning**
Seasonal forecast by SKU category (immunity/respiratory peak in winter, digestive/cooling peak in summer) blended with historical channel-wise sell-through

**5.6 Regulatory & Compliance**
Batch documentation, AYUSH license renewal, GST filing, and traceability from raw material lot to finished batch to distributor invoice

---

## 6. ERP Modules (Conceptual — for data modeling purposes)

Since Balaji Pharma does not currently run a formal ERP (per its founder-led, Excel-driven SME profile), this project will simulate the **data structures a lightweight ERP/MIS would produce**, mapped to standard modules:

| Module | Purpose | Key Tables to Model |
|---|---|---|
| **Procurement (MM)** | Vendor master, PO, GRN | Vendors, Purchase Orders, Goods Receipts, Raw Material Master |
| **Production (PP)** | Batch planning & execution | Bill of Materials (BOM), Production Orders, Batch Master |
| **Inventory (IM/WM)** | Stock tracking | Raw Material Stock, Finished Goods Stock, Stock Ledger/Movements |
| **Sales & Distribution (SD)** | Order management | Customer Master (Distributors/Retailers), Sales Orders, Invoices, Channel Master |
| **Quality Management (QM)** | Batch QC | QC Test Results, Batch Release Records |
| **Finance (FI)** | Payments & revenue | Vendor Payments, Customer Receivables, GL summary (simplified) |

This mirrors Utkarsh's own SAP BODS/ECC background — the synthetic data model will resemble a scaled-down SAP-style schema, which also makes the portfolio project credible to a technical reviewer.

---

## 7. Data Sources (to be generated in later phases)

1. **Vendor Master & Purchase Orders** — 7–8 suppliers, lead times, pricing, payment terms
2. **Raw Material Inventory** — herb/oil/excipient stock, batches, expiry
3. **Bill of Materials (BOM)** — per-SKU formulation recipe
4. **Production/Batch Records** — batch size, yield, QC pass/fail, production calendar
5. **Finished Goods Inventory** — stock levels, movements, FEFO tracking
6. **Customer/Channel Master** — Distributors (2), Retail outlets (~150), Amazon, IndiaMART, OTC
7. **Sales Orders & Invoices** — channel-wise, SKU-wise, date-stamped transactions (2018–2025/26)
8. **Returns/Expiry Write-offs** — realistic for pharma/Ayurvedic shelf-life products
9. **Pricing & Discount Master** — channel-based pricing (distributor cost price vs Amazon MRP vs OTC price)
10. **Financial Summary** — revenue, COGS, gross margin by month/year

---

## 8. KPIs

**Supply Chain & Operations**
- Order Fill Rate (%) by channel
- Inventory Turnover Ratio (raw material & finished goods, separately)
- Stockout frequency by SKU
- Batch Yield % (actual output vs planned)
- Supplier On-Time Delivery %
- Raw Material Lead Time (avg, by supplier category)
- Finished Goods Days of Inventory
- Expiry/Wastage % of production

**Sales & Distribution**
- Revenue by Channel (Distributor/Retail/Amazon/IndiaMART/OTC)
- SKU-wise contribution to revenue (Pareto/ABC analysis)
- Outlet productivity (revenue per outlet, per market rep)
- YoY Growth Rate (overall and by channel)
- Seasonal Demand Index by SKU category

**Financial**
- Gross Margin % by channel and SKU
- Working Capital Days (Inventory + Receivables − Payables)
- Cost of Goods Sold (COGS) trend

**Compliance/Quality**
- Batch QC Pass Rate
- Time-to-Batch-Release (days from production to release)

---

## 9. Reporting Structure

| Report | Audience | Frequency |
|---|---|---|
| Daily Dispatch & Order Status | Sales Ops / Founder | Daily |
| Weekly Production & Batch Report | Production Supervisor / Founder | Weekly |
| Monthly Inventory Health (Raw + FG) | Founder / Procurement | Monthly |
| Monthly Channel Sales Performance | Founder | Monthly |
| Quarterly Vendor Performance Review | Procurement | Quarterly |
| Annual P&L and Growth Review | Founder | Annually |

**Power BI Dashboard Layers (planned):**
1. Executive Overview (Revenue, Growth, Margin — Founder-facing)
2. Supply Chain Ops Dashboard (Inventory, Procurement, Batch KPIs)
3. Sales & Channel Performance Dashboard
4. Quality/Compliance Tracker

---

## 10. Future Analytics Roadmap

**Phase 1 — Foundational Data Layer (this project's next step)**
Generate consistent master data (vendors, SKUs, BOM, customers/channels) and multi-year transactional data (2018–2025) with realistic seasonality, growth, and constraints.

**Phase 2 — Descriptive Analytics**
Power BI dashboards covering sales trends, inventory health, and vendor performance.

**Phase 3 — Diagnostic Analytics**
Root-cause views: why stockouts happen, which SKUs drive margin erosion, which channels are most volatile.

**Phase 4 — Predictive Analytics**
Demand forecasting by SKU/season, inventory reorder point optimization, supplier lead-time risk modeling.

**Phase 5 — Prescriptive/Advanced**
Production planning optimization, working capital optimization, what-if scenario modeling (e.g., new outlet expansion, new SKU launch impact).

---

### Next Step
With this business definition locked in, the next phase will generate the **Master Data Layer** — Vendor Master, SKU/Product Master, BOM, and Customer/Channel Master — which all future transactional data (2018–2025) will reference for referential integrity.
