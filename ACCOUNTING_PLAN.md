# Street Voices Accounting — Ultimate Implementation Plan

> Modeled after [GnuCash](https://github.com/Gnucash/gnucash), the gold standard of open-source accounting. Adapted for the Street Voices web platform.

---

## Architecture Overview

```
Browser (localhost:3180/accounting)
    |
    +-- accounting.js          (SPA controller, routing, UI)
    +-- /sbapi/v1/accounting/* (REST API on nanobot-api)
    +-- PostgreSQL             (persistent storage via vectordb)
```

**Stack:** Vanilla JS frontend (matches existing SV pattern) + Python/FastAPI backend + PostgreSQL

---

## Phase 1 — Core Engine (Double-Entry Bookkeeping)

### 1.1 Chart of Accounts

The foundation of all accounting. Every transaction debits one account and credits another.

- [ ] Account tree (hierarchical parent/child structure)
- [ ] Account types:
  - **Assets:** Cash, Bank, Accounts Receivable, Investments, Fixed Assets, Other Assets
  - **Liabilities:** Credit Card, Accounts Payable, Loans, Other Liabilities
  - **Equity:** Opening Balance, Retained Earnings, Owner's Equity
  - **Income:** Salary, Interest, Grants, Donations, Other Income
  - **Expenses:** Rent, Utilities, Payroll, Software, Hardware, Travel, Marketing, Legal, Insurance, Supplies, Depreciation, Misc
- [ ] Default chart of accounts templates (Nonprofit, Small Business, Personal, Freelancer)
- [ ] Custom account creation with: name, code, description, type, currency, parent, placeholder flag, hidden flag, tax-related flag, notes
- [ ] Account search and filtering
- [ ] Account renumbering and reordering
- [ ] Placeholder accounts (grouping-only, no direct transactions)
- [ ] Opening balances setup wizard

### 1.2 Double-Entry Transaction Engine

- [ ] Every transaction has balanced debits and credits (sum must equal zero)
- [ ] Split transactions (one transaction across multiple accounts)
- [ ] Transaction fields: date, description/memo, splits (account + debit/credit amount), number/reference, reconciliation status, notes
- [ ] Auto-completion for account names and descriptions
- [ ] Transaction templates / quick-fill from previous entries
- [ ] Duplicate transaction detection
- [ ] Void transactions (preserve audit trail, don't delete)
- [ ] Reversing entries
- [ ] Transaction linking (relate invoice payments to invoices)
- [ ] Batch transaction entry

### 1.3 General Ledger / Register

- [ ] Account register view (per-account transaction list)
- [ ] General ledger view (all transactions across accounts)
- [ ] Column configuration: date, num, description, transfer account, deposit, withdrawal, balance
- [ ] Running balance calculation
- [ ] Sort by any column
- [ ] Filter by date range, amount range, description, memo, reconciliation status
- [ ] Split view (expand to see all splits of a transaction)
- [ ] Inline editing in register
- [ ] Color coding by transaction type

---

## Phase 2 — Reconciliation & Bank Integration

### 2.1 Account Reconciliation

- [ ] Reconciliation wizard: enter statement date and ending balance
- [ ] Check off cleared transactions
- [ ] Auto-match imported transactions with existing entries
- [ ] Reconciliation difference display (target vs actual)
- [ ] Save partial reconciliation (resume later)
- [ ] Reconciliation report (list of reconciled items)
- [ ] Reconciliation status per transaction: **n** (new), **c** (cleared), **y** (reconciled)

### 2.2 Import / Export

- [ ] **CSV import** with column mapping wizard
- [ ] **OFX/QFX import** (Open Financial Exchange — bank downloads)
- [ ] **QIF import** (Quicken Interchange Format)
- [ ] **CSV export** (transactions, accounts, reports)
- [ ] **JSON export/import** (full backup/restore)
- [ ] **PDF export** for all reports
- [ ] Import duplicate detection and merge
- [ ] Import rules (auto-categorize by description patterns)
- [ ] Bank statement import matching algorithm

---

## Phase 3 — Financial Reports & Statements

### 3.1 Standard Reports

- [ ] **Profit & Loss (Income Statement)** — revenue minus expenses for a period
- [ ] **Balance Sheet** — assets, liabilities, equity at a point in time
- [ ] **Cash Flow Statement** — cash inflows and outflows
- [ ] **Trial Balance** — all account balances to verify debits = credits
- [ ] **General Journal** — chronological list of all entries
- [ ] **Account Summary** — balances for all accounts
- [ ] **Transaction Report** — filtered/sorted transaction listing

### 3.2 Income / Expense Reports

- [ ] Income by category (bar chart + table)
- [ ] Expense by category (pie chart + table)
- [ ] Income vs Expense comparison (multi-period bar chart)
- [ ] Income/Expense over time (line chart — monthly/quarterly/yearly)
- [ ] Category breakdown with percentage of total

### 3.3 Asset / Liability Reports

- [ ] Net worth over time
- [ ] Asset allocation breakdown
- [ ] Liability summary
- [ ] Account balance tracker (line chart per account)

### 3.4 Budget Reports

- [ ] Budget vs Actual comparison
- [ ] Budget variance report (over/under per category)
- [ ] Budget flow (projected vs actual over time)

### 3.5 Report Features

- [ ] Date range selector (custom, this month, this quarter, this year, last year, YTD)
- [ ] Comparison periods (this year vs last year)
- [ ] Drill-down from report to underlying transactions
- [ ] Report customization (select accounts, columns, grouping)
- [ ] Save report configurations as presets
- [ ] Print-friendly layouts
- [ ] Chart types: bar, line, pie, stacked bar, area

---

## Phase 4 — Business Features (AR / AP)

### 4.1 Customers & Accounts Receivable

- [ ] Customer database: name, contact info, billing address, shipping address, tax ID, terms, credit limit, notes
- [ ] Customer search and listing
- [ ] **Invoices:**
  - Create, edit, duplicate, void
  - Line items with description, quantity, unit price, discount, tax
  - Invoice numbering (auto-increment, custom prefix)
  - Due date calculation from payment terms
  - Invoice status: Draft, Sent, Viewed, Partial, Paid, Overdue, Void
  - Attachments (receipts, contracts)
  - Notes and internal memos
- [ ] **Payments received:**
  - Apply payment to one or multiple invoices
  - Partial payments
  - Overpayment handling (credit balance)
  - Payment methods (cash, check, bank transfer, card)
- [ ] **Credit notes** (refunds / adjustments)
- [ ] **Aging report:** Current, 30, 60, 90, 90+ days
- [ ] Customer statement generation
- [ ] Recurring invoices (auto-generate on schedule)
- [ ] Late payment reminders

### 4.2 Vendors & Accounts Payable

- [ ] Vendor database: name, contact info, address, tax ID, payment terms, notes
- [ ] **Bills:**
  - Enter, edit, duplicate, void
  - Line items with description, quantity, unit price, tax
  - Bill numbering
  - Due date tracking
  - Bill status: Draft, Received, Partial, Paid, Overdue, Void
- [ ] **Bill payments:**
  - Pay one or multiple bills
  - Partial payments
  - Payment scheduling
- [ ] **Vendor credit notes**
- [ ] **AP Aging report:** Current, 30, 60, 90, 90+ days
- [ ] Purchase orders (optional)
- [ ] 1099 vendor tracking (US tax)

### 4.3 Employees

- [ ] Employee database: name, contact info, ID, hire date, department, pay rate
- [ ] Expense vouchers / reimbursements
- [ ] Employee expense reports
- [ ] Time tracking integration (hours worked)

---

## Phase 5 — Budgeting

### 5.1 Budget Management

- [ ] Create budgets by fiscal year or custom period
- [ ] Budget per account (income and expense accounts)
- [ ] Monthly/quarterly/annual budget amounts
- [ ] Auto-fill from prior period actuals
- [ ] Multiple budget scenarios (optimistic, conservative, etc.)
- [ ] Budget notes per category
- [ ] Copy budget to new period

### 5.2 Budget Monitoring

- [ ] Real-time budget vs actual dashboard
- [ ] Percentage used per category
- [ ] Visual progress bars
- [ ] Over-budget alerts / warnings
- [ ] Remaining budget projections
- [ ] Burn rate calculations

---

## Phase 6 — Scheduled & Recurring Transactions

- [ ] Scheduled transaction definitions:
  - Frequency: daily, weekly, biweekly, monthly, quarterly, annually, custom
  - Start date, end date (or indefinite)
  - Auto-create or prompt for review
  - Reminder days before due
- [ ] Upcoming transactions calendar view
- [ ] Scheduled transaction review queue (approve/skip/edit)
- [ ] Cash flow forecasting based on scheduled transactions
- [ ] Mortgage/loan amortization schedules (auto-generate payment series)
- [ ] Payroll schedule

---

## Phase 7 — Multi-Currency & Commodities

### 7.1 Multi-Currency

- [ ] Multiple currency support per account
- [ ] Exchange rate database (manual entry + auto-fetch)
- [ ] Transaction-level exchange rates
- [ ] Currency conversion on reports
- [ ] Unrealized gains/losses calculation
- [ ] Base/reporting currency setting

### 7.2 Investments & Securities

- [ ] Stock/mutual fund accounts
- [ ] Buy, sell, dividend, split, return of capital transactions
- [ ] Lot tracking (FIFO, LIFO, average cost, specific identification)
- [ ] Capital gains/losses calculation
- [ ] Price database (historical prices)
- [ ] Online price quotes (auto-fetch stock prices)
- [ ] Portfolio summary report
- [ ] Investment performance report (ROI, annualized returns)

---

## Phase 8 — Tax Features

- [ ] Tax-related account flagging
- [ ] Tax category/code assignments per account
- [ ] Tax report (income, deductions, credits grouped by tax form/line)
- [ ] Sales tax tracking:
  - Tax rates per jurisdiction
  - Tax collected on invoices
  - Tax paid on purchases
  - Tax liability report
  - Tax return summary
- [ ] Tax year settings
- [ ] TXF export (for TurboTax, TaxCut import)
- [ ] 1099 report (vendor payments over threshold)
- [ ] Withholding tax tracking

---

## Phase 9 — Dashboard & Analytics

### 9.1 Main Dashboard

- [ ] Summary cards: total assets, total liabilities, net worth, income (MTD), expenses (MTD), net income
- [ ] Income vs Expenses chart (last 12 months)
- [ ] Top expense categories (pie chart)
- [ ] Cash balance trend line
- [ ] Accounts receivable aging summary
- [ ] Accounts payable aging summary
- [ ] Upcoming scheduled transactions
- [ ] Recent transactions list
- [ ] Bank account balances at a glance

### 9.2 KPI Widgets

- [ ] Current ratio (current assets / current liabilities)
- [ ] Quick ratio
- [ ] Debt-to-equity ratio
- [ ] Gross profit margin
- [ ] Net profit margin
- [ ] Operating expense ratio
- [ ] Days sales outstanding (DSO)
- [ ] Days payable outstanding (DPO)

---

## Phase 10 — Loans & Mortgages

- [ ] Loan account setup: principal, interest rate, term, payment frequency, start date
- [ ] Amortization schedule generation
- [ ] Payment breakdown: principal vs interest per payment
- [ ] Extra payment handling
- [ ] Payoff calculator
- [ ] Refinance comparison
- [ ] Multiple loan tracking
- [ ] Loan balance over time chart

---

## Phase 11 — Settings & Administration

### 11.1 Company / Organization Settings

- [ ] Organization name, address, logo
- [ ] Fiscal year (calendar year, custom start month)
- [ ] Default currency
- [ ] Number format preferences
- [ ] Date format preferences

### 11.2 User Management

- [ ] Role-based access: Admin, Accountant, Viewer, Invoice-only
- [ ] Audit log (who changed what, when)
- [ ] User preferences per user

### 11.3 Data Management

- [ ] Full backup/restore (JSON)
- [ ] Data export (all accounts + transactions)
- [ ] Account closing (year-end close)
- [ ] Fiscal year rollover
- [ ] Data integrity check (verify debits = credits)
- [ ] Archive old transactions

---

## Phase 12 — Nonprofit / Grant-Specific Features

> Tailored for Street Voices as a nonprofit platform.

- [ ] Fund/grant tracking (restricted vs unrestricted)
- [ ] Grant budget tracking (per grant)
- [ ] Grant expense allocation
- [ ] Grant reporting templates
- [ ] Donor management
- [ ] In-kind donation tracking
- [ ] Program vs administrative expense allocation
- [ ] Form 990 data preparation
- [ ] Functional expense allocation (program, management, fundraising)

---

## Technical Implementation Plan

### Backend API Endpoints (nanobot-api)

```
POST   /sbapi/v1/accounting/accounts          Create account
GET    /sbapi/v1/accounting/accounts           List accounts (tree)
GET    /sbapi/v1/accounting/accounts/:id       Get account details
PUT    /sbapi/v1/accounting/accounts/:id       Update account
DELETE /sbapi/v1/accounting/accounts/:id       Delete account (if no txns)

POST   /sbapi/v1/accounting/transactions       Create transaction
GET    /sbapi/v1/accounting/transactions       List/search transactions
GET    /sbapi/v1/accounting/transactions/:id   Get transaction + splits
PUT    /sbapi/v1/accounting/transactions/:id   Update transaction
DELETE /sbapi/v1/accounting/transactions/:id   Void transaction

POST   /sbapi/v1/accounting/customers          Create customer
GET    /sbapi/v1/accounting/customers          List customers
POST   /sbapi/v1/accounting/invoices           Create invoice
GET    /sbapi/v1/accounting/invoices           List invoices
PUT    /sbapi/v1/accounting/invoices/:id       Update invoice
POST   /sbapi/v1/accounting/invoices/:id/pay   Apply payment

POST   /sbapi/v1/accounting/vendors            Create vendor
GET    /sbapi/v1/accounting/vendors            List vendors
POST   /sbapi/v1/accounting/bills              Create bill
POST   /sbapi/v1/accounting/bills/:id/pay      Pay bill

GET    /sbapi/v1/accounting/reports/:type       Generate report
POST   /sbapi/v1/accounting/budgets            Create budget
GET    /sbapi/v1/accounting/budgets            List budgets

POST   /sbapi/v1/accounting/reconcile          Start reconciliation
PUT    /sbapi/v1/accounting/reconcile/:id      Update reconciliation
POST   /sbapi/v1/accounting/import             Import file (CSV/OFX/QIF)
GET    /sbapi/v1/accounting/export/:format     Export data
```

### Database Schema (PostgreSQL)

```sql
-- Core tables
acct_accounts        (id, parent_id, name, code, type, currency, description,
                      placeholder, hidden, tax_related, notes, created_at)
acct_transactions    (id, date, num, description, notes, currency, created_at, voided)
acct_splits          (id, transaction_id, account_id, memo, amount, quantity,
                      reconcile_state, reconcile_date)

-- Business tables
acct_customers       (id, name, contact, billing_addr, shipping_addr, tax_id,
                      terms, credit_limit, notes, active)
acct_vendors         (id, name, contact, address, tax_id, terms, notes, active)
acct_employees       (id, name, contact, hire_date, department, pay_rate, active)
acct_invoices        (id, customer_id, date, due_date, number, status, notes)
acct_invoice_items   (id, invoice_id, description, quantity, unit_price, discount,
                      tax_rate, account_id)
acct_bills           (id, vendor_id, date, due_date, number, status, notes)
acct_bill_items      (id, bill_id, description, quantity, unit_price, tax_rate,
                      account_id)

-- Budget tables
acct_budgets         (id, name, fiscal_year, notes, created_at)
acct_budget_amounts  (id, budget_id, account_id, period, amount)

-- Scheduled transactions
acct_scheduled_txns  (id, name, frequency, start_date, end_date, last_run,
                      next_run, auto_create, template_transaction_id)

-- Multi-currency
acct_prices          (id, commodity, currency, date, source, price)

-- Tax
acct_tax_codes       (id, code, description, form, line)
acct_tax_assignments (id, account_id, tax_code_id)

-- Reconciliation
acct_reconcile_sessions (id, account_id, statement_date, ending_balance,
                         status, created_at)

-- Import rules
acct_import_rules    (id, pattern, account_id, description_override, priority)

-- Audit
acct_audit_log       (id, user, action, entity_type, entity_id, changes, timestamp)
```

### Frontend Module Structure

```
deploy/custom-ui/
  accounting.js              Main SPA controller + routing
  accounting-accounts.js     Chart of accounts module
  accounting-register.js     Transaction register/ledger
  accounting-reconcile.js    Reconciliation wizard
  accounting-invoices.js     AR: invoices + payments
  accounting-bills.js        AP: bills + payments
  accounting-reports.js      Reports engine + charts
  accounting-budgets.js      Budget management
  accounting-dashboard.js    Dashboard widgets
  accounting-import.js       Import/export wizard
  accounting-settings.js     Settings + admin
  accounting-styles.css      All accounting CSS
  images/sidebar-icons/
    accounting.svg            Sidebar icon
```

---

## Implementation Priority

| Priority | Phase | Effort | Description |
|----------|-------|--------|-------------|
| P0 | 1 | 2 weeks | Core engine: chart of accounts, double-entry, register |
| P0 | 3.1 | 1 week | Basic reports: P&L, Balance Sheet, Trial Balance |
| P0 | 9.1 | 1 week | Dashboard with summary cards and charts |
| P1 | 2 | 1 week | Reconciliation + CSV/OFX import |
| P1 | 4.1 | 2 weeks | Invoicing + AR |
| P1 | 4.2 | 1 week | Bills + AP |
| P1 | 5 | 1 week | Budgeting |
| P2 | 6 | 1 week | Scheduled/recurring transactions |
| P2 | 3.2-3.5 | 1 week | Advanced reports + charts |
| P2 | 8 | 1 week | Tax features + sales tax |
| P2 | 12 | 1 week | Nonprofit/grant tracking |
| P3 | 7 | 1 week | Multi-currency + investments |
| P3 | 10 | 3 days | Loans & mortgages |
| P3 | 11 | 3 days | Settings & admin |

**Total estimated: ~16 weeks for full implementation**

---

## Design Principles

1. **Double-entry or nothing** — every transaction balances. No single-entry shortcuts.
2. **Audit trail** — never delete, only void. Log every change.
3. **Offline-first** — localStorage fallback when API is unavailable.
4. **Match the SV aesthetic** — dark theme, monospace accents, clean cards.
5. **Non-accountant friendly** — plain language, guided wizards, smart defaults.
6. **GnuCash-compatible export** — enable migration path via CSV/OFX.
