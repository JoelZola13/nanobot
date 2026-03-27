/**
 * Accounting page — financial management with transactions, categories, and reporting.
 *
 * Features: dashboard summary, transaction CRUD, filtering, category management,
 * CSV export. Data persisted to localStorage.
 */

window.AccountingPage = {
  _transactions: [],
  _categories: [
    'Revenue', 'Services', 'Grants', 'Donations',
    'Payroll', 'Rent', 'Utilities', 'Software', 'Hardware',
    'Travel', 'Marketing', 'Legal', 'Insurance', 'Misc',
  ],
  _filter: 'all',
  _sortCol: 'date',
  _sortAsc: false,
  _editingId: null,
  _unsubs: [],
  _STORAGE_KEY: 'nanobot:accounting:transactions',
  _CAT_STORAGE_KEY: 'nanobot:accounting:categories',

  render(container) {
    container.innerHTML = `
      <div class="page">
        <div class="page-header">
          <span class="page-title">Accounting</span>
          <div class="acct-header-actions">
            <button class="btn btn-primary" id="acct-add-btn">+ transaction</button>
            <button class="btn acct-btn-export" id="acct-export-btn">export csv</button>
          </div>
        </div>
        <div class="page-body">
          <div id="acct-summary"></div>
          <div id="acct-form-area"></div>
          <div class="acct-toolbar" id="acct-toolbar">
            <div class="acct-filters">
              <button class="acct-filter-btn active" data-filter="all">All</button>
              <button class="acct-filter-btn" data-filter="income">Income</button>
              <button class="acct-filter-btn" data-filter="expense">Expense</button>
            </div>
            <div class="acct-search-wrap">
              <input type="text" id="acct-search" class="acct-search" placeholder="search transactions...">
            </div>
          </div>
          <div id="acct-table-area"></div>
        </div>
      </div>
    `;

    this._loadData();
    this._bindEvents();
    this._renderSummary();
    this._renderTable();
  },

  destroy() {
    this._unsubs.forEach(fn => fn());
    this._unsubs = [];
    this._editingId = null;
  },

  // ── Data Persistence ──

  _loadData() {
    try {
      const raw = localStorage.getItem(this._STORAGE_KEY);
      this._transactions = raw ? JSON.parse(raw) : this._seedData();
    } catch {
      this._transactions = this._seedData();
    }
    try {
      const cats = localStorage.getItem(this._CAT_STORAGE_KEY);
      if (cats) this._categories = JSON.parse(cats);
    } catch { /* keep defaults */ }
  },

  _saveData() {
    localStorage.setItem(this._STORAGE_KEY, JSON.stringify(this._transactions));
    localStorage.setItem(this._CAT_STORAGE_KEY, JSON.stringify(this._categories));
  },

  _seedData() {
    const seed = [
      { id: this._uid(), date: '2026-03-01', description: 'Client invoice #1042', category: 'Revenue', type: 'income', amount: 5200.00 },
      { id: this._uid(), date: '2026-03-03', description: 'Cloud hosting - March', category: 'Software', type: 'expense', amount: 349.99 },
      { id: this._uid(), date: '2026-03-05', description: 'Office rent', category: 'Rent', type: 'expense', amount: 2100.00 },
      { id: this._uid(), date: '2026-03-10', description: 'Consulting fee - Project Alpha', category: 'Services', type: 'income', amount: 3750.00 },
      { id: this._uid(), date: '2026-03-12', description: 'Team payroll - biweekly', category: 'Payroll', type: 'expense', amount: 8500.00 },
      { id: this._uid(), date: '2026-03-15', description: 'Grant disbursement Q1', category: 'Grants', type: 'income', amount: 10000.00 },
      { id: this._uid(), date: '2026-03-18', description: 'Internet & phone', category: 'Utilities', type: 'expense', amount: 189.50 },
      { id: this._uid(), date: '2026-03-20', description: 'Marketing campaign', category: 'Marketing', type: 'expense', amount: 1200.00 },
      { id: this._uid(), date: '2026-03-22', description: 'Hardware - monitors x3', category: 'Hardware', type: 'expense', amount: 1350.00 },
      { id: this._uid(), date: '2026-03-25', description: 'Client invoice #1043', category: 'Revenue', type: 'income', amount: 4800.00 },
    ];
    localStorage.setItem(this._STORAGE_KEY, JSON.stringify(seed));
    return seed;
  },

  // ── Events ──

  _bindEvents() {
    const addBtn = document.getElementById('acct-add-btn');
    addBtn?.addEventListener('click', () => this._showForm());

    const exportBtn = document.getElementById('acct-export-btn');
    exportBtn?.addEventListener('click', () => this._exportCSV());

    // Filter buttons
    document.querySelectorAll('.acct-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.acct-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this._filter = btn.dataset.filter;
        this._renderTable();
      });
    });

    // Search
    const search = document.getElementById('acct-search');
    search?.addEventListener('input', () => this._renderTable());
  },

  // ── Summary Cards ──

  _renderSummary() {
    const container = document.getElementById('acct-summary');
    if (!container) return;

    const income = this._transactions
      .filter(t => t.type === 'income')
      .reduce((sum, t) => sum + t.amount, 0);
    const expense = this._transactions
      .filter(t => t.type === 'expense')
      .reduce((sum, t) => sum + t.amount, 0);
    const net = income - expense;
    const count = this._transactions.length;

    container.innerHTML = `
      <div class="acct-summary-grid">
        <div class="acct-summary-card">
          <div class="acct-summary-label">Total Income</div>
          <div class="acct-summary-value income">$${this._fmt(income)}</div>
        </div>
        <div class="acct-summary-card">
          <div class="acct-summary-label">Total Expenses</div>
          <div class="acct-summary-value expense">$${this._fmt(expense)}</div>
        </div>
        <div class="acct-summary-card">
          <div class="acct-summary-label">Net Balance</div>
          <div class="acct-summary-value ${net >= 0 ? 'positive' : 'negative'}">$${this._fmt(Math.abs(net))}${net < 0 ? ' deficit' : ''}</div>
        </div>
        <div class="acct-summary-card">
          <div class="acct-summary-label">Transactions</div>
          <div class="acct-summary-value neutral">${count}</div>
        </div>
      </div>
    `;
  },

  // ── Transaction Form ──

  _showForm(txn) {
    const area = document.getElementById('acct-form-area');
    if (!area) return;

    this._editingId = txn ? txn.id : null;
    const isEdit = !!txn;

    const catOptions = this._categories.map(c =>
      `<option value="${this._esc(c)}" ${txn && txn.category === c ? 'selected' : ''}>${this._esc(c)}</option>`
    ).join('');

    area.innerHTML = `
      <div class="acct-form">
        <div class="acct-form-title">${isEdit ? 'Edit' : 'New'} Transaction</div>
        <div class="acct-form-grid">
          <div class="acct-form-field">
            <label>Date</label>
            <input type="date" id="acct-f-date" value="${txn ? txn.date : new Date().toISOString().slice(0, 10)}">
          </div>
          <div class="acct-form-field">
            <label>Type</label>
            <select id="acct-f-type">
              <option value="income" ${txn && txn.type === 'income' ? 'selected' : ''}>Income</option>
              <option value="expense" ${txn && txn.type === 'expense' ? 'selected' : ''}>Expense</option>
            </select>
          </div>
          <div class="acct-form-field">
            <label>Category</label>
            <select id="acct-f-category">${catOptions}</select>
          </div>
          <div class="acct-form-field">
            <label>Amount ($)</label>
            <input type="number" id="acct-f-amount" step="0.01" min="0" placeholder="0.00" value="${txn ? txn.amount : ''}">
          </div>
          <div class="acct-form-field acct-form-field-wide">
            <label>Description</label>
            <input type="text" id="acct-f-desc" placeholder="Transaction description" value="${txn ? this._esc(txn.description) : ''}">
          </div>
        </div>
        <div class="acct-form-actions">
          <button class="btn btn-primary" id="acct-f-save">${isEdit ? 'update' : 'save'}</button>
          <button class="btn acct-btn-cancel" id="acct-f-cancel">cancel</button>
        </div>
      </div>
    `;

    document.getElementById('acct-f-save').addEventListener('click', () => this._saveTransaction());
    document.getElementById('acct-f-cancel').addEventListener('click', () => this._hideForm());
    document.getElementById('acct-f-desc').focus();
  },

  _hideForm() {
    const area = document.getElementById('acct-form-area');
    if (area) area.innerHTML = '';
    this._editingId = null;
  },

  _saveTransaction() {
    const date = document.getElementById('acct-f-date')?.value;
    const type = document.getElementById('acct-f-type')?.value;
    const category = document.getElementById('acct-f-category')?.value;
    const amount = parseFloat(document.getElementById('acct-f-amount')?.value);
    const description = document.getElementById('acct-f-desc')?.value?.trim();

    if (!date || !description || isNaN(amount) || amount <= 0) {
      window.app.toast('Please fill all fields with valid values', 'error');
      return;
    }

    if (this._editingId) {
      const idx = this._transactions.findIndex(t => t.id === this._editingId);
      if (idx >= 0) {
        this._transactions[idx] = { ...this._transactions[idx], date, type, category, amount, description };
      }
    } else {
      this._transactions.unshift({ id: this._uid(), date, type, category, amount, description });
    }

    this._saveData();
    this._hideForm();
    this._renderSummary();
    this._renderTable();
    window.app.toast(this._editingId ? 'Transaction updated' : 'Transaction added', 'success');
  },

  // ── Transaction Table ──

  _renderTable() {
    const container = document.getElementById('acct-table-area');
    if (!container) return;

    const search = (document.getElementById('acct-search')?.value || '').toLowerCase();

    let rows = this._transactions.filter(t => {
      if (this._filter !== 'all' && t.type !== this._filter) return false;
      if (search && !t.description.toLowerCase().includes(search) && !t.category.toLowerCase().includes(search)) return false;
      return true;
    });

    // Sort
    rows.sort((a, b) => {
      let cmp = 0;
      if (this._sortCol === 'date') cmp = a.date.localeCompare(b.date);
      else if (this._sortCol === 'amount') cmp = a.amount - b.amount;
      else if (this._sortCol === 'description') cmp = a.description.localeCompare(b.description);
      else if (this._sortCol === 'category') cmp = a.category.localeCompare(b.category);
      return this._sortAsc ? cmp : -cmp;
    });

    if (rows.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">◎</div>
          <div class="empty-state-text">No transactions found. Add one to get started.</div>
        </div>
      `;
      return;
    }

    const sortIcon = (col) => this._sortCol === col ? (this._sortAsc ? ' ↑' : ' ↓') : '';

    const tableRows = rows.map(t => `
      <tr data-id="${t.id}">
        <td>${this._esc(t.date)}</td>
        <td>
          <span class="badge ${t.type === 'income' ? 'badge-green' : 'badge-red'}">${t.type}</span>
        </td>
        <td><span class="acct-cat-pill">${this._esc(t.category)}</span></td>
        <td>${this._esc(t.description)}</td>
        <td class="acct-amount ${t.type}">
          ${t.type === 'income' ? '+' : '-'}$${this._fmt(t.amount)}
        </td>
        <td class="acct-actions-cell">
          <button class="acct-row-btn acct-edit-btn" data-id="${t.id}" title="Edit">✎</button>
          <button class="acct-row-btn acct-del-btn" data-id="${t.id}" title="Delete">✕</button>
        </td>
      </tr>
    `).join('');

    container.innerHTML = `
      <div class="acct-table-wrap">
        <table class="data-table acct-table">
          <thead>
            <tr>
              <th class="acct-sortable" data-col="date">Date${sortIcon('date')}</th>
              <th>Type</th>
              <th class="acct-sortable" data-col="category">Category${sortIcon('category')}</th>
              <th class="acct-sortable" data-col="description">Description${sortIcon('description')}</th>
              <th class="acct-sortable" data-col="amount">Amount${sortIcon('amount')}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    `;

    // Sortable headers
    container.querySelectorAll('.acct-sortable').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.col;
        if (this._sortCol === col) {
          this._sortAsc = !this._sortAsc;
        } else {
          this._sortCol = col;
          this._sortAsc = true;
        }
        this._renderTable();
      });
    });

    // Edit buttons
    container.querySelectorAll('.acct-edit-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const txn = this._transactions.find(t => t.id === btn.dataset.id);
        if (txn) this._showForm(txn);
      });
    });

    // Delete buttons
    container.querySelectorAll('.acct-del-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this._transactions = this._transactions.filter(t => t.id !== btn.dataset.id);
        this._saveData();
        this._renderSummary();
        this._renderTable();
        window.app.toast('Transaction deleted', 'info');
      });
    });
  },

  // ── Export ──

  _exportCSV() {
    if (this._transactions.length === 0) {
      window.app.toast('No transactions to export', 'error');
      return;
    }
    const header = 'Date,Type,Category,Description,Amount';
    const rows = this._transactions.map(t =>
      `${t.date},${t.type},"${t.category}","${t.description.replace(/"/g, '""')}",${t.type === 'income' ? '' : '-'}${t.amount.toFixed(2)}`
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nanobot-accounting-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    window.app.toast('CSV exported', 'success');
  },

  // ── Helpers ──

  _uid() {
    return 'txn_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  },

  _fmt(n) {
    return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  },

  _esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
};
