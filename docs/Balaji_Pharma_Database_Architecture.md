# Balaji Pharma — Supply Chain Analytics Database Architecture

**Prepared as:** Senior Data Architect deliverable
**Purpose:** Foundational relational schema to support Sales, Inventory, Demand Forecasting, Procurement, Manufacturing, Finance, and Distribution analytics — in SQL, Python, Power BI, and ML pipelines.
**Design standard:** Third Normal Form (3NF) for operational integrity, structured so it rolls up cleanly into Power BI star schemas.

---

## 0. Business Context & Assumptions

- Balaji Pharma manufactures Ayurvedic OTC products (tablets, syrups, churna, oils, capsules) and distributes them via 2 distributors to 150+ retail outlets in Indore.
- Operations begin **August 2018**; ramp-up through 2019–2020, steady-state with seasonality through **July 2026** (~8 years).
- Seasonal demand: immunity/respiratory SKUs peak Oct–Feb; digestive SKUs peak around festival season; skin/hair care has a mild summer bump.
- Batch manufacturing (not continuous flow), finite weekly capacity, herbal raw materials carry longer/variable lead times (7–21 days) vs. packaging (3–10 days).
- All monetary fields in INR; sales/purchase invoicing includes GST fields.
- This document defines **structure only** — no data rows are generated here.

---

## 1. Schema Overview (Table Groups)

| Domain | Tables |
|---|---|
| Core Dimensions | `dim_date`, `dim_product`, `dim_product_category`, `dim_raw_material`, `dim_raw_material_category`, `dim_supplier`, `dim_customer`, `dim_distributor`, `dim_employee`, `dim_warehouse` |
| Manufacturing | `bill_of_materials`, `production_batch`, `production_batch_consumption`, `production_batch_output`, `quality_check` |
| Procurement | `purchase_order`, `purchase_order_line`, `goods_receipt`, `goods_receipt_line`, `supplier_invoice`, `supplier_payment` |
| Inventory | `raw_material_inventory_txn`, `finished_goods_inventory_txn`, `raw_material_stock_snapshot`, `finished_goods_stock_snapshot` |
| Sales & Distribution | `sales_order`, `sales_order_line`, `distribution_shipment`, `shipment_line`, `sales_invoice`, `customer_payment` |
| Finance | `expense`, `chart_of_accounts` |
| Forecasting / ML | `demand_forecast`, `forecast_accuracy_log` |

Total: **31 tables**.

---

## 2. Core Dimension Tables

### 2.1 `dim_date`
- **PK:** `date_key` (INT, format YYYYMMDD)
- **FK:** none
- **Columns:** `full_date` DATE, `day_of_week` TINYINT, `day_name` VARCHAR(10), `month` TINYINT, `month_name` VARCHAR(10), `quarter` TINYINT, `year` SMALLINT, `is_weekend` BOOLEAN, `is_festival_season` BOOLEAN, `fiscal_year` VARCHAR(9), `fiscal_quarter` TINYINT
- **Relationships:** referenced by every fact table needing time intelligence
- **Business Purpose:** enables seasonality analysis, YoY/MoM trending, fiscal reporting (Apr–Mar per Indian FY)
- **Expected Rows:** ~2,920 (Aug 2018–Jul 2026 daily)

### 2.2 `dim_product_category`
- **PK:** `category_id` INT
- **Columns:** `category_name` VARCHAR(50) (e.g., Digestive Care, Immunity, Respiratory, Skin Care, Pain Relief, Women's Health, General Wellness, Hair Care), `therapeutic_segment` VARCHAR(50), `shelf_life_months` SMALLINT
- **Business Purpose:** groups SKUs for category-level sales/margin analysis
- **Expected Rows:** 8

### 2.3 `dim_product`
- **PK:** `product_id` INT
- **FK:** `category_id` → `dim_product_category`
- **Columns:** `product_code` VARCHAR(15), `product_name` VARCHAR(100), `dosage_form` VARCHAR(20) (Tablet/Syrup/Churna/Oil/Capsule), `pack_size` VARCHAR(20), `unit_of_measure` VARCHAR(10), `mrp` DECIMAL(10,2), `standard_cost` DECIMAL(10,2), `launch_date` DATE, `discontinued_date` DATE NULL, `min_batch_size` INT, `is_active` BOOLEAN
- **Relationships:** one-to-many with `bill_of_materials`, `sales_order_line`, `production_batch`
- **Business Purpose:** master SKU catalog driving BOM, sales, and inventory
- **Expected Rows:** 45

### 2.4 `dim_raw_material_category`
- **PK:** `rm_category_id` INT
- **Columns:** `category_name` VARCHAR(50) (Herbal Extract, Excipient, Primary Packaging, Secondary Packaging, Solvent, Preservative)
- **Business Purpose:** groups raw materials for procurement spend analysis
- **Expected Rows:** 6

### 2.5 `dim_raw_material`
- **PK:** `raw_material_id` INT
- **FK:** `rm_category_id` → `dim_raw_material_category`
- **Columns:** `material_code` VARCHAR(15), `material_name` VARCHAR(100), `unit_of_measure` VARCHAR(10), `standard_unit_cost` DECIMAL(10,2), `min_stock_level` DECIMAL(10,2), `max_stock_level` DECIMAL(10,2), `reorder_point` DECIMAL(10,2), `default_lead_time_days` SMALLINT, `is_seasonal_availability` BOOLEAN
- **Relationships:** many-to-many with `dim_product` via `bill_of_materials`; one-to-many with `purchase_order_line`
- **Business Purpose:** raw material master for procurement & MRP logic
- **Expected Rows:** 80

### 2.6 `dim_supplier`
- **PK:** `supplier_id` INT
- **Columns:** `supplier_name` VARCHAR(100), `supplier_type` VARCHAR(30) (Herbal Raw Material / Packaging / Excipient), `city` VARCHAR(50), `onboarded_date` DATE, `avg_lead_time_days` SMALLINT, `reliability_score` DECIMAL(3,2) (0–1, derived from on-time delivery history), `payment_terms_days` SMALLINT, `gstin` VARCHAR(15), `is_active` BOOLEAN
- **Business Purpose:** vendor master for procurement analytics and supplier scorecards
- **Expected Rows:** 8

### 2.7 `dim_customer`
- **PK:** `customer_id` INT
- **Columns:** `customer_name` VARCHAR(100), `outlet_type` VARCHAR(30) (Pharmacy/General Store/Ayurvedic Store), `area` VARCHAR(50) (Indore locality), `onboarded_date` DATE, `churned_date` DATE NULL, `credit_limit` DECIMAL(10,2), `payment_terms_days` SMALLINT, `assigned_rep_id` INT (FK → `dim_employee`), `distributor_id` INT (FK → `dim_distributor`), `is_active` BOOLEAN
- **Relationships:** one-to-many with `sales_order`
- **Business Purpose:** retail outlet master; supports growth-curve and churn analysis (0 → 150+ outlets over time)
- **Expected Rows:** ~180 (accounts for onboarding + some churn over 8 years)

### 2.8 `dim_distributor`
- **PK:** `distributor_id` INT
- **Columns:** `distributor_name` VARCHAR(100), `region_covered` VARCHAR(50), `onboarded_date` DATE, `commission_pct` DECIMAL(4,2)
- **Business Purpose:** models the 2-distributor channel structure
- **Expected Rows:** 2

### 2.9 `dim_employee`
- **PK:** `employee_id` INT
- **Columns:** `employee_name` VARCHAR(100), `role` VARCHAR(30) (Founder/GM, Market Representative, Production Staff, Distributor Coordinator), `join_date` DATE, `exit_date` DATE NULL, `monthly_salary` DECIMAL(10,2), `department` VARCHAR(30)
- **Relationships:** referenced by `dim_customer.assigned_rep_id`, `production_batch.supervisor_id`
- **Business Purpose:** headcount-linked cost modeling; sales-rep performance analysis
- **Expected Rows:** ~20 (lean team growth from 5 → 13+)

### 2.10 `dim_warehouse`
- **PK:** `warehouse_id` INT
- **Columns:** `warehouse_name` VARCHAR(50), `warehouse_type` VARCHAR(20) (Raw Material Store / Finished Goods Store / Distribution Hub), `city` VARCHAR(50), `capacity_units` INT
- **Business Purpose:** location dimension for inventory and distribution tables
- **Expected Rows:** 3

---

## 3. Manufacturing Tables

### 3.1 `bill_of_materials` (BOM)
- **PK:** `bom_id` INT
- **FK:** `product_id` → `dim_product`, `raw_material_id` → `dim_raw_material`
- **Columns:** `quantity_per_unit` DECIMAL(10,4), `uom` VARCHAR(10), `is_critical_component` BOOLEAN
- **Relationships:** many-to-many bridge between `dim_product` and `dim_raw_material`
- **Business Purpose:** defines material requirements per finished unit — drives MRP, batch costing, and procurement forecasting
- **Expected Rows:** ~225 (45 products × ~5 materials avg)

### 3.2 `production_batch`
- **PK:** `batch_id` INT
- **FK:** `product_id` → `dim_product`, `warehouse_id` → `dim_warehouse`, `supervisor_id` → `dim_employee`
- **Columns:** `batch_number` VARCHAR(20), `planned_start_date` DATE, `actual_start_date` DATE, `actual_end_date` DATE, `planned_quantity` INT, `actual_quantity_produced` INT, `batch_status` VARCHAR(20) (Planned/In-Progress/Completed/QA-Hold/Rejected), `manufacturing_cost` DECIMAL(12,2)
- **Relationships:** one-to-many with `production_batch_consumption`, `production_batch_output`, `quality_check`
- **Business Purpose:** core manufacturing execution record — capacity utilization, yield analysis, cost tracking
- **Expected Rows:** ~1,500 (averaging ~3-4 batches/week across active years)

### 3.3 `production_batch_consumption`
- **PK:** `consumption_id` INT
- **FK:** `batch_id` → `production_batch`, `raw_material_id` → `dim_raw_material`
- **Columns:** `quantity_consumed` DECIMAL(10,4), `uom` VARCHAR(10), `unit_cost_at_consumption` DECIMAL(10,2)
- **Business Purpose:** actual raw material draw-down per batch — feeds inventory depletion and true batch costing (vs. BOM standard)
- **Expected Rows:** ~7,500 (1,500 batches × ~5 materials)

### 3.4 `production_batch_output`
- **PK:** `output_id` INT
- **FK:** `batch_id` → `production_batch` (1:1 typically, occasionally split for partial QA pass)
- **Columns:** `quantity_good` INT, `quantity_rejected` INT, `output_date` DATE, `expiry_date` DATE, `warehouse_id` INT (FK → `dim_warehouse`)
- **Business Purpose:** links completed production into finished goods inventory
- **Expected Rows:** ~1,600

### 3.5 `quality_check`
- **PK:** `qc_id` INT
- **FK:** `batch_id` → `production_batch`
- **Columns:** `check_date` DATE, `parameter_tested` VARCHAR(50), `result` VARCHAR(20) (Pass/Fail), `remarks` VARCHAR(200), `inspector_id` INT (FK → `dim_employee`)
- **Business Purpose:** compliance/QA tracking — supports regulatory compliance analytics referenced in resume
- **Expected Rows:** ~1,500 (roughly 1 per batch)

---

## 4. Procurement Tables

### 4.1 `purchase_order`
- **PK:** `po_id` INT
- **FK:** `supplier_id` → `dim_supplier`
- **Columns:** `po_number` VARCHAR(20), `order_date` DATE, `expected_delivery_date` DATE, `po_status` VARCHAR(20) (Open/Partially Received/Closed/Cancelled), `total_amount` DECIMAL(12,2)
- **Relationships:** one-to-many with `purchase_order_line`
- **Business Purpose:** procurement transaction header — supplier spend, PO cycle time
- **Expected Rows:** ~600 (8 suppliers × ~10 orders/year × ~7.5 active years)

### 4.2 `purchase_order_line`
- **PK:** `po_line_id` INT
- **FK:** `po_id` → `purchase_order`, `raw_material_id` → `dim_raw_material`
- **Columns:** `quantity_ordered` DECIMAL(10,2), `unit_price` DECIMAL(10,2), `line_amount` DECIMAL(12,2), `quantity_received` DECIMAL(10,2)
- **Business Purpose:** line-level procurement detail for material-level spend and lead-time variance analysis
- **Expected Rows:** ~1,800

### 4.3 `goods_receipt`
- **PK:** `grn_id` INT
- **FK:** `po_id` → `purchase_order`, `warehouse_id` → `dim_warehouse`
- **Columns:** `grn_number` VARCHAR(20), `receipt_date` DATE, `received_by` INT (FK → `dim_employee`)
- **Business Purpose:** header for actual receipt event — supports on-time delivery / lead-time-actual-vs-planned analytics
- **Expected Rows:** ~650 (some POs received in multiple shipments)

### 4.4 `goods_receipt_line`
- **PK:** `grn_line_id` INT
- **FK:** `grn_id` → `goods_receipt`, `po_line_id` → `purchase_order_line`
- **Columns:** `quantity_received` DECIMAL(10,2), `quality_status` VARCHAR(20) (Accepted/Rejected/Partial), `batch_reference` VARCHAR(20)
- **Business Purpose:** ties physical receipt to PO line, feeds inventory transaction table
- **Expected Rows:** ~1,900

### 4.5 `supplier_invoice`
- **PK:** `supplier_invoice_id` INT
- **FK:** `po_id` → `purchase_order`, `supplier_id` → `dim_supplier`
- **Columns:** `invoice_number` VARCHAR(20), `invoice_date` DATE, `due_date` DATE, `invoice_amount` DECIMAL(12,2), `gst_amount` DECIMAL(10,2), `payment_status` VARCHAR(20)
- **Business Purpose:** AP tracking, cash flow forecasting
- **Expected Rows:** ~600

### 4.6 `supplier_payment`
- **PK:** `payment_id` INT
- **FK:** `supplier_invoice_id` → `supplier_invoice`
- **Columns:** `payment_date` DATE, `amount_paid` DECIMAL(12,2), `payment_mode` VARCHAR(20) (NEFT/Cheque/UPI)
- **Business Purpose:** AP cash outflow tracking, DPO (Days Payable Outstanding) calculation
- **Expected Rows:** ~650 (some invoices paid in installments)

---

## 5. Inventory Tables

### 5.1 `raw_material_inventory_txn`
- **PK:** `txn_id` BIGINT
- **FK:** `raw_material_id` → `dim_raw_material`, `warehouse_id` → `dim_warehouse`
- **Columns:** `txn_date` DATE, `txn_type` VARCHAR(20) (Receipt/Consumption/Adjustment/Return), `quantity` DECIMAL(10,2) (+/-), `reference_type` VARCHAR(20) (PO/Batch/Manual), `reference_id` INT, `running_balance` DECIMAL(10,2)
- **Business Purpose:** immutable ledger of all raw material movement — single source of truth for stock-on-hand
- **Expected Rows:** ~10,000

### 5.2 `finished_goods_inventory_txn`
- **PK:** `txn_id` BIGINT
- **FK:** `product_id` → `dim_product`, `warehouse_id` → `dim_warehouse`
- **Columns:** `txn_date` DATE, `txn_type` VARCHAR(20) (Production/Sale/Return/Adjustment/Expiry-Writeoff), `quantity` DECIMAL(10,2) (+/-), `reference_type` VARCHAR(20), `reference_id` INT, `running_balance` DECIMAL(10,2)
- **Business Purpose:** immutable ledger of finished goods movement — feeds stockout and days-of-cover analysis
- **Expected Rows:** ~30,000

### 5.3 `raw_material_stock_snapshot`
- **PK:** `snapshot_id` BIGINT
- **FK:** `raw_material_id` → `dim_raw_material`, `warehouse_id` → `dim_warehouse`, `date_key` → `dim_date`
- **Columns:** `closing_quantity` DECIMAL(10,2), `closing_value` DECIMAL(12,2)
- **Business Purpose:** month-end (or weekly) stock snapshot for fast BI reporting without re-aggregating the full ledger
- **Expected Rows:** ~80 materials × ~96 months ≈ 7,680

### 5.4 `finished_goods_stock_snapshot`
- **PK:** `snapshot_id` BIGINT
- **FK:** `product_id` → `dim_product`, `warehouse_id` → `dim_warehouse`, `date_key` → `dim_date`
- **Columns:** `closing_quantity` DECIMAL(10,2), `closing_value` DECIMAL(12,2)
- **Business Purpose:** enables fast trend reporting on finished goods inventory and expiry risk
- **Expected Rows:** ~45 products × ~96 months ≈ 4,320

---

## 6. Sales & Distribution Tables

### 6.1 `sales_order`
- **PK:** `sales_order_id` BIGINT
- **FK:** `customer_id` → `dim_customer`, `distributor_id` → `dim_distributor`
- **Columns:** `order_number` VARCHAR(20), `order_date` DATE, `requested_delivery_date` DATE, `order_status` VARCHAR(20) (Open/Fulfilled/Partially Fulfilled/Cancelled), `total_amount` DECIMAL(12,2)
- **Relationships:** one-to-many with `sales_order_line`
- **Business Purpose:** core demand transaction — feeds sales analytics and demand forecasting
- **Expected Rows:** ~25,000 (growth-weighted across 150+ outlets over ~7.5 active years)

### 6.2 `sales_order_line`
- **PK:** `sales_order_line_id` BIGINT
- **FK:** `sales_order_id` → `sales_order`, `product_id` → `dim_product`
- **Columns:** `quantity_ordered` INT, `unit_price` DECIMAL(10,2), `discount_pct` DECIMAL(4,2), `line_amount` DECIMAL(12,2), `quantity_fulfilled` INT
- **Business Purpose:** SKU-level demand signal — primary input for demand forecasting models
- **Expected Rows:** ~95,000 (avg ~3.8 lines per order)

### 6.3 `distribution_shipment`
- **PK:** `shipment_id` BIGINT
- **FK:** `distributor_id` → `dim_distributor`, `warehouse_id` → `dim_warehouse`
- **Columns:** `shipment_number` VARCHAR(20), `dispatch_date` DATE, `delivery_date` DATE, `vehicle_type` VARCHAR(20), `shipment_status` VARCHAR(20)
- **Business Purpose:** groups multiple sales orders into a delivery run — supports logistics cost & on-time delivery analytics
- **Expected Rows:** ~6,000

### 6.4 `shipment_line`
- **PK:** `shipment_line_id` BIGINT
- **FK:** `shipment_id` → `distribution_shipment`, `sales_order_id` → `sales_order`
- **Columns:** `quantity_shipped` INT
- **Business Purpose:** links shipments to the orders they fulfill (many orders can share one shipment run)
- **Expected Rows:** ~26,000

### 6.5 `sales_invoice`
- **PK:** `invoice_id` BIGINT
- **FK:** `sales_order_id` → `sales_order`, `customer_id` → `dim_customer`
- **Columns:** `invoice_number` VARCHAR(20), `invoice_date` DATE, `due_date` DATE, `invoice_amount` DECIMAL(12,2), `gst_amount` DECIMAL(10,2), `payment_status` VARCHAR(20)
- **Business Purpose:** AR tracking, revenue recognition
- **Expected Rows:** ~25,000

### 6.6 `customer_payment`
- **PK:** `payment_id` BIGINT
- **FK:** `invoice_id` → `sales_invoice`
- **Columns:** `payment_date` DATE, `amount_paid` DECIMAL(12,2), `payment_mode` VARCHAR(20)
- **Business Purpose:** AR cash inflow, DSO (Days Sales Outstanding) calculation, credit risk analysis by outlet
- **Expected Rows:** ~27,000

---

## 7. Finance Tables

### 7.1 `chart_of_accounts`
- **PK:** `account_id` INT
- **Columns:** `account_name` VARCHAR(50), `account_type` VARCHAR(20) (Revenue/COGS/Opex/Asset/Liability)
- **Business Purpose:** classification backbone for expense/finance reporting
- **Expected Rows:** ~25

### 7.2 `expense`
- **PK:** `expense_id` INT
- **FK:** `account_id` → `chart_of_accounts`, `date_key` → `dim_date`
- **Columns:** `expense_date` DATE, `category` VARCHAR(30) (Salaries/Rent/Utilities/Marketing/Logistics/Compliance), `amount` DECIMAL(12,2), `notes` VARCHAR(200)
- **Business Purpose:** operating cost tracking — required for P&L and margin analytics (ties to resume's "full P&L ownership")
- **Expected Rows:** ~700 (monthly recurring × categories × ~96 months)

---

## 8. Forecasting / ML Tables

### 8.1 `demand_forecast`
- **PK:** `forecast_id` BIGINT
- **FK:** `product_id` → `dim_product`, `date_key` → `dim_date` (month-level)
- **Columns:** `forecast_month` DATE, `forecasted_quantity` DECIMAL(10,2), `forecast_method` VARCHAR(30) (Moving Avg/ARIMA/Prophet/ML-Regression), `generated_date` DATE, `model_version` VARCHAR(20)
- **Business Purpose:** stores model output for comparison against actuals — supports demand planning and MRP
- **Expected Rows:** ~45 products × 96 months ≈ 4,320

### 8.2 `forecast_accuracy_log`
- **PK:** `log_id` BIGINT
- **FK:** `forecast_id` → `demand_forecast`
- **Columns:** `actual_quantity` DECIMAL(10,2), `absolute_error` DECIMAL(10,2), `mape` DECIMAL(6,4), `evaluated_date` DATE
- **Business Purpose:** tracks forecast model performance over time (MAPE/bias) — supports continuous ML model improvement
- **Expected Rows:** ~4,320 (one per evaluated forecast)

---

## 9. Entity Relationship Summary (Key Flows)

```
dim_supplier ──< purchase_order ──< purchase_order_line >── dim_raw_material
                       │                                          │
                       └──< goods_receipt ──< goods_receipt_line ─┘
                                                                    │
dim_raw_material ──< raw_material_inventory_txn                    │
      │                                                             │
      └──< bill_of_materials >── dim_product                        │
                    │                                               │
                    └──< production_batch ──< production_batch_consumption
                                │                    (draws from raw material stock)
                                ├──< production_batch_output ──> finished_goods_inventory_txn
                                └──< quality_check

dim_customer ──< sales_order ──< sales_order_line >── dim_product
      │                │
      │                └──< sales_invoice ──< customer_payment
dim_distributor ──< distribution_shipment ──< shipment_line >── sales_order

dim_product ──< demand_forecast ──< forecast_accuracy_log
```

---

## 10. Design Notes for Realistic Data Generation (Next Phase)

When populating this schema, the following business logic must be enforced (not randomized):

1. **Growth curve**: `dim_customer.onboarded_date` should follow an S-curve — slow in 2018–2019, accelerating through 2020, plateauing near 150–180 outlets by 2021 onward.
2. **Seasonality**: `sales_order_line.quantity_ordered` for Respiratory/Immunity category should show Oct–Feb uplift; Digestive category should spike ±2 weeks around major festivals (Diwali, Navratri).
3. **Lead time realism**: `purchase_order.expected_delivery_date` minus `order_date` should match `dim_raw_material.default_lead_time_days`, with herbal extracts showing more variance than packaging.
4. **Inventory constraint**: `production_batch` cannot be created unless `raw_material_inventory_txn` running balance covers the BOM requirement — this enforces referential/business consistency between procurement, inventory, and manufacturing.
5. **Capacity constraint**: total planned production per week should not exceed a defined plant capacity ceiling (derived from `dim_employee` headcount in Production department).
6. **Financial consistency**: `sales_invoice.invoice_amount` must equal the sum of related `sales_order_line.line_amount`, and `customer_payment` totals should lag invoice dates by realistic `payment_terms_days` (with some late payers, reflecting real SME credit risk).

---

**Total tables:** 31 | **Total dimension tables:** 10 | **Total fact/transaction tables:** 21

Next step: generate synthetic data batch-by-batch (dimensions first, then procurement → manufacturing → inventory → sales → finance, in that dependency order) to preserve referential integrity.
