/**
 * Street Voices — Accounting Module
 * Full-page accounting app injected when visiting /accounting.
 * Handles transactions, categories, dashboard summary, and CSV export.
 * Data persists to localStorage.
 */
(function () {
  'use strict';

  var PAGE_ID = 'sv-accounting-page';
  var STORAGE_KEY = 'sv:accounting:transactions';
  var CAT_STORAGE_KEY = 'sv:accounting:categories';

  var DEFAULT_CATEGORIES = [
    'Revenue', 'Services', 'Grants', 'Donations',
    'Payroll', 'Rent', 'Utilities', 'Software', 'Hardware',
    'Travel', 'Marketing', 'Legal', 'Insurance', 'Misc',
  ];

  var state = {
    transactions: [],
    categories: DEFAULT_CATEGORIES.slice(),
    filter: 'all',
    sortCol: 'date',
    sortAsc: false,
    editingId: null,
    search: '',
  };

  function uid() {
    return 'txn_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  function fmt(n) {
    return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function esc(str) {
    var div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  function loadData() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      state.transactions = raw ? JSON.parse(raw) : seedData();
    } catch (e) {
      state.transactions = seedData();
    }
    try {
      var cats = localStorage.getItem(CAT_STORAGE_KEY);
      if (cats) state.categories = JSON.parse(cats);
    } catch (e) { /* keep defaults */ }
  }

  function saveData() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.transactions));
    localStorage.setItem(CAT_STORAGE_KEY, JSON.stringify(state.categories));
  }

  function seedData() {
    var seed = [
      { id: uid(), date: '2026-03-01', description: 'Client invoice #1042', category: 'Revenue', type: 'income', amount: 5200.00 },
      { id: uid(), date: '2026-03-03', description: 'Cloud hosting - March', category: 'Software', type: 'expense', amount: 349.99 },
      { id: uid(), date: '2026-03-05', description: 'Office rent', category: 'Rent', type: 'expense', amount: 2100.00 },
      { id: uid(), date: '2026-03-10', description: 'Consulting fee - Project Alpha', category: 'Services', type: 'income', amount: 3750.00 },
      { id: uid(), date: '2026-03-12', description: 'Team payroll - biweekly', category: 'Payroll', type: 'expense', amount: 8500.00 },
      { id: uid(), date: '2026-03-15', description: 'Grant disbursement Q1', category: 'Grants', type: 'income', amount: 10000.00 },
      { id: uid(), date: '2026-03-18', description: 'Internet & phone', category: 'Utilities', type: 'expense', amount: 189.50 },
      { id: uid(), date: '2026-03-20', description: 'Marketing campaign', category: 'Marketing', type: 'expense', amount: 1200.00 },
      { id: uid(), date: '2026-03-22', description: 'Hardware - monitors x3', category: 'Hardware', type: 'expense', amount: 1350.00 },
      { id: uid(), date: '2026-03-25', description: 'Client invoice #1043', category: 'Revenue', type: 'income', amount: 4800.00 },
    ];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seed));
    return seed;
  }

  // ── Styles ──

  function injectStyles() {
    if (document.getElementById('sv-acct-styles')) return;
    var style = document.createElement('style');
    style.id = 'sv-acct-styles';
    style.textContent = [
      '#' + PAGE_ID + ' {',
      '  position: fixed; left: 260px; top: 0; right: 0; bottom: 0; z-index: 100;',
      '  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;',
      '  font-size: 14px; overflow-y: auto; display: flex; flex-direction: column;',
      '}',
      '.dark #' + PAGE_ID + ' { background: #0d0d0d; color: #e8e8ed; }',
      'html:not(.dark) #' + PAGE_ID + ' { background: #fafafa; color: #1a1a2e; }',
      '',
      '#' + PAGE_ID + ' .acct-topbar {',
      '  position: sticky; top: 0; z-index: 10; padding: 16px 24px;',
      '  display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;',
      '  border-bottom: 1px solid rgba(128,128,128,0.15);',
      '}',
      '.dark #' + PAGE_ID + ' .acct-topbar { background: rgba(13,13,13,0.95); backdrop-filter: blur(12px); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-topbar { background: rgba(250,250,250,0.95); backdrop-filter: blur(12px); }',
      '#' + PAGE_ID + ' .acct-topbar-title { font-size: 18px; font-weight: 700; }',
      '.dark #' + PAGE_ID + ' .acct-topbar-title { color: #e8e8ed; }',
      '#' + PAGE_ID + ' .acct-topbar-actions { display: flex; gap: 8px; }',
      '',
      '#' + PAGE_ID + ' .acct-btn {',
      '  font-family: inherit; font-size: 13px; font-weight: 500; border: none; border-radius: 8px;',
      '  padding: 8px 16px; cursor: pointer; transition: all 0.15s;',
      '}',
      '.dark #' + PAGE_ID + ' .acct-btn-primary { background: #22c55e; color: #000; }',
      '.dark #' + PAGE_ID + ' .acct-btn-primary:hover { background: #16a34a; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-btn-primary { background: #16a34a; color: #fff; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-btn-primary:hover { background: #15803d; }',
      '.dark #' + PAGE_ID + ' .acct-btn-secondary { background: #1c1c22; color: #8888a0; border: 1px solid #2a2a3a; }',
      '.dark #' + PAGE_ID + ' .acct-btn-secondary:hover { background: #2a2a3a; color: #e8e8ed; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-btn-secondary { background: #fff; color: #555; border: 1px solid #ddd; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-btn-secondary:hover { background: #f0f0f0; color: #1a1a2e; }',
      '',
      '#' + PAGE_ID + ' .acct-body { flex: 1; padding: 20px 24px; }',
      '',
      '/* Summary Cards */',
      '#' + PAGE_ID + ' .acct-summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; margin-bottom: 24px; }',
      '#' + PAGE_ID + ' .acct-card { border-radius: 12px; padding: 18px 20px; }',
      '.dark #' + PAGE_ID + ' .acct-card { background: #1c1c22; border: 1px solid rgba(255,255,255,0.06); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-card { background: #fff; border: 1px solid #e5e5e5; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }',
      '#' + PAGE_ID + ' .acct-card-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }',
      '.dark #' + PAGE_ID + ' .acct-card-label { color: #6b7280; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-card-label { color: #9ca3af; }',
      '#' + PAGE_ID + ' .acct-card-value { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }',
      '#' + PAGE_ID + ' .acct-card-value.income { color: #22c55e; }',
      '#' + PAGE_ID + ' .acct-card-value.expense { color: #ef4444; }',
      '#' + PAGE_ID + ' .acct-card-value.positive { color: #22c55e; }',
      '#' + PAGE_ID + ' .acct-card-value.negative { color: #ef4444; }',
      '#' + PAGE_ID + ' .acct-card-value.neutral { color: #3b82f6; }',
      '',
      '/* Toolbar */',
      '#' + PAGE_ID + ' .acct-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }',
      '#' + PAGE_ID + ' .acct-filters { display: flex; gap: 4px; }',
      '#' + PAGE_ID + ' .acct-filter-btn { font-family: inherit; font-size: 12px; font-weight: 500; border-radius: 6px; padding: 6px 14px; cursor: pointer; transition: all 0.15s; }',
      '.dark #' + PAGE_ID + ' .acct-filter-btn { background: #1c1c22; border: 1px solid #2a2a3a; color: #8888a0; }',
      '.dark #' + PAGE_ID + ' .acct-filter-btn:hover { background: #2a2a3a; color: #e8e8ed; }',
      '.dark #' + PAGE_ID + ' .acct-filter-btn.active { background: rgba(34,197,94,0.15); border-color: rgba(34,197,94,0.3); color: #22c55e; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-filter-btn { background: #fff; border: 1px solid #ddd; color: #666; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-filter-btn:hover { background: #f5f5f5; color: #333; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-filter-btn.active { background: rgba(22,163,74,0.1); border-color: rgba(22,163,74,0.3); color: #16a34a; }',
      '#' + PAGE_ID + ' .acct-search { font-family: inherit; font-size: 13px; border-radius: 6px; padding: 7px 12px; width: 240px; outline: none; transition: border-color 0.15s; }',
      '.dark #' + PAGE_ID + ' .acct-search { background: #1c1c22; border: 1px solid #2a2a3a; color: #e8e8ed; }',
      '.dark #' + PAGE_ID + ' .acct-search:focus { border-color: #3b82f6; }',
      '.dark #' + PAGE_ID + ' .acct-search::placeholder { color: #555; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-search { background: #fff; border: 1px solid #ddd; color: #1a1a2e; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-search:focus { border-color: #3b82f6; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-search::placeholder { color: #aaa; }',
      '',
      '/* Form */',
      '#' + PAGE_ID + ' .acct-form { border-radius: 12px; padding: 18px 20px; margin-bottom: 20px; }',
      '.dark #' + PAGE_ID + ' .acct-form { background: #1c1c22; border: 1px solid rgba(255,255,255,0.06); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-form { background: #fff; border: 1px solid #e5e5e5; }',
      '#' + PAGE_ID + ' .acct-form-title { font-size: 14px; font-weight: 600; margin-bottom: 14px; }',
      '#' + PAGE_ID + ' .acct-form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px; }',
      '#' + PAGE_ID + ' .acct-form-field-wide { grid-column: 1 / -1; }',
      '#' + PAGE_ID + ' .acct-form-field label { display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }',
      '.dark #' + PAGE_ID + ' .acct-form-field label { color: #6b7280; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-form-field label { color: #9ca3af; }',
      '#' + PAGE_ID + ' .acct-form-field input, #' + PAGE_ID + ' .acct-form-field select { width: 100%; font-family: inherit; font-size: 13px; border-radius: 6px; padding: 8px 10px; outline: none; transition: border-color 0.15s; box-sizing: border-box; }',
      '.dark #' + PAGE_ID + ' .acct-form-field input, .dark #' + PAGE_ID + ' .acct-form-field select { background: #0d0d0d; border: 1px solid #2a2a3a; color: #e8e8ed; }',
      '.dark #' + PAGE_ID + ' .acct-form-field input:focus, .dark #' + PAGE_ID + ' .acct-form-field select:focus { border-color: #3b82f6; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-form-field input, html:not(.dark) #' + PAGE_ID + ' .acct-form-field select { background: #fafafa; border: 1px solid #ddd; color: #1a1a2e; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-form-field input:focus, html:not(.dark) #' + PAGE_ID + ' .acct-form-field select:focus { border-color: #3b82f6; }',
      '#' + PAGE_ID + ' .acct-form-actions { display: flex; gap: 8px; }',
      '',
      '/* Table */',
      '#' + PAGE_ID + ' .acct-table-wrap { overflow-x: auto; border-radius: 12px; }',
      '.dark #' + PAGE_ID + ' .acct-table-wrap { background: #1c1c22; border: 1px solid rgba(255,255,255,0.06); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-table-wrap { background: #fff; border: 1px solid #e5e5e5; }',
      '#' + PAGE_ID + ' .acct-table { width: 100%; border-collapse: collapse; font-size: 13px; }',
      '#' + PAGE_ID + ' .acct-table th { text-align: left; padding: 10px 14px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }',
      '.dark #' + PAGE_ID + ' .acct-table th { color: #555; border-bottom: 1px solid rgba(255,255,255,0.06); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-table th { color: #9ca3af; border-bottom: 1px solid #e5e5e5; }',
      '#' + PAGE_ID + ' .acct-table td { padding: 10px 14px; }',
      '.dark #' + PAGE_ID + ' .acct-table td { color: #c0c0cc; border-bottom: 1px solid rgba(255,255,255,0.04); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-table td { color: #4a4a5a; border-bottom: 1px solid #f0f0f0; }',
      '.dark #' + PAGE_ID + ' .acct-table tr:hover td { background: rgba(255,255,255,0.03); }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-table tr:hover td { background: #f9f9fb; }',
      '#' + PAGE_ID + ' .acct-sortable { cursor: pointer; user-select: none; }',
      '.dark #' + PAGE_ID + ' .acct-sortable:hover { color: #3b82f6; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-sortable:hover { color: #2563eb; }',
      '#' + PAGE_ID + ' .acct-amount { font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }',
      '#' + PAGE_ID + ' .acct-amount.income { color: #22c55e; }',
      '#' + PAGE_ID + ' .acct-amount.expense { color: #ef4444; }',
      '#' + PAGE_ID + ' .acct-badge { display: inline-block; font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 4px; }',
      '.dark #' + PAGE_ID + ' .acct-badge-green { background: rgba(34,197,94,0.15); color: #22c55e; }',
      '.dark #' + PAGE_ID + ' .acct-badge-red { background: rgba(239,68,68,0.15); color: #ef4444; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-badge-green { background: rgba(22,163,74,0.1); color: #16a34a; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-badge-red { background: rgba(220,38,38,0.1); color: #dc2626; }',
      '#' + PAGE_ID + ' .acct-cat-pill { font-size: 11px; padding: 2px 8px; border-radius: 4px; }',
      '.dark #' + PAGE_ID + ' .acct-cat-pill { background: rgba(255,255,255,0.06); color: #8888a0; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-cat-pill { background: #f0f0f5; color: #666; }',
      '#' + PAGE_ID + ' .acct-row-btn { background: none; border: none; cursor: pointer; font-size: 13px; padding: 4px 6px; border-radius: 4px; transition: all 0.15s; }',
      '.dark #' + PAGE_ID + ' .acct-row-btn { color: #555; }',
      '.dark #' + PAGE_ID + ' .acct-row-btn:hover { background: rgba(255,255,255,0.06); color: #e8e8ed; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-row-btn { color: #aaa; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-row-btn:hover { background: #f0f0f5; color: #333; }',
      '.dark #' + PAGE_ID + ' .acct-del-btn:hover { color: #ef4444; }',
      '.dark #' + PAGE_ID + ' .acct-edit-btn:hover { color: #3b82f6; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-del-btn:hover { color: #dc2626; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-edit-btn:hover { color: #2563eb; }',
      '',
      '/* Empty state */',
      '#' + PAGE_ID + ' .acct-empty { display: flex; flex-direction: column; align-items: center; padding: 48px 20px; text-align: center; }',
      '.dark #' + PAGE_ID + ' .acct-empty { color: #555; }',
      'html:not(.dark) #' + PAGE_ID + ' .acct-empty { color: #aaa; }',
      '#' + PAGE_ID + ' .acct-empty-icon { font-size: 36px; margin-bottom: 12px; opacity: 0.4; }',
      '',
      '/* Toast */',
      '#sv-acct-toast { position: fixed; bottom: 20px; right: 20px; z-index: 10000; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size: 13px; padding: 10px 18px; border-radius: 8px; transform: translateY(8px); opacity: 0; transition: all 0.2s; pointer-events: none; }',
      '#sv-acct-toast.visible { transform: translateY(0); opacity: 1; }',
      '.dark #sv-acct-toast { background: #1c1c22; color: #e8e8ed; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }',
      'html:not(.dark) #sv-acct-toast { background: #fff; color: #333; border: 1px solid #e5e5e5; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }',
      '',
      '/* Responsive */',
      '@media (max-width: 768px) {',
      '  #' + PAGE_ID + ' { left: 0; }',
      '  #' + PAGE_ID + ' .acct-body { padding: 12px 16px; }',
      '  #' + PAGE_ID + ' .acct-summary { grid-template-columns: repeat(2, 1fr); }',
      '}',
    ].join('\n');
    document.head.appendChild(style);
  }

  // ── Toast ──

  var toastTimer = null;
  function toast(msg) {
    var el = document.getElementById('sv-acct-toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'sv-acct-toast';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    if (toastTimer) clearTimeout(toastTimer);
    requestAnimationFrame(function () { el.classList.add('visible'); });
    toastTimer = setTimeout(function () {
      el.classList.remove('visible');
    }, 2500);
  }

  // ── Render ──

  function render() {
    var page = document.getElementById(PAGE_ID);
    if (!page) return;

    var income = 0, expense = 0;
    state.transactions.forEach(function (t) {
      if (t.type === 'income') income += t.amount;
      else expense += t.amount;
    });
    var net = income - expense;

    page.innerHTML =
      '<div class="acct-topbar">' +
        '<div class="acct-topbar-title">Accounting</div>' +
        '<div class="acct-topbar-actions">' +
          '<button class="acct-btn acct-btn-primary" id="acct-add-btn">+ Transaction</button>' +
          '<button class="acct-btn acct-btn-secondary" id="acct-export-btn">Export CSV</button>' +
        '</div>' +
      '</div>' +
      '<div class="acct-body">' +
        '<div class="acct-summary">' +
          '<div class="acct-card"><div class="acct-card-label">Total Income</div><div class="acct-card-value income">$' + fmt(income) + '</div></div>' +
          '<div class="acct-card"><div class="acct-card-label">Total Expenses</div><div class="acct-card-value expense">$' + fmt(expense) + '</div></div>' +
          '<div class="acct-card"><div class="acct-card-label">Net Balance</div><div class="acct-card-value ' + (net >= 0 ? 'positive' : 'negative') + '">$' + fmt(Math.abs(net)) + (net < 0 ? ' deficit' : '') + '</div></div>' +
          '<div class="acct-card"><div class="acct-card-label">Transactions</div><div class="acct-card-value neutral">' + state.transactions.length + '</div></div>' +
        '</div>' +
        '<div id="acct-form-area"></div>' +
        renderToolbar() +
        '<div id="acct-table-area">' + renderTable() + '</div>' +
      '</div>';

    bindEvents();
  }

  function renderToolbar() {
    var filters = ['all', 'income', 'expense'];
    var btns = filters.map(function (f) {
      return '<button class="acct-filter-btn' + (state.filter === f ? ' active' : '') + '" data-filter="' + f + '">' + f.charAt(0).toUpperCase() + f.slice(1) + '</button>';
    }).join('');

    return '<div class="acct-toolbar">' +
      '<div class="acct-filters">' + btns + '</div>' +
      '<input type="text" class="acct-search" id="acct-search" placeholder="Search transactions..." value="' + esc(state.search) + '">' +
    '</div>';
  }

  function renderTable() {
    var rows = state.transactions.filter(function (t) {
      if (state.filter !== 'all' && t.type !== state.filter) return false;
      if (state.search) {
        var s = state.search.toLowerCase();
        if (t.description.toLowerCase().indexOf(s) === -1 && t.category.toLowerCase().indexOf(s) === -1) return false;
      }
      return true;
    });

    rows.sort(function (a, b) {
      var cmp = 0;
      if (state.sortCol === 'date') cmp = a.date.localeCompare(b.date);
      else if (state.sortCol === 'amount') cmp = a.amount - b.amount;
      else if (state.sortCol === 'description') cmp = a.description.localeCompare(b.description);
      else if (state.sortCol === 'category') cmp = a.category.localeCompare(b.category);
      return state.sortAsc ? cmp : -cmp;
    });

    if (rows.length === 0) {
      return '<div class="acct-empty"><div class="acct-empty-icon">&#9678;</div><div>No transactions found. Add one to get started.</div></div>';
    }

    var sortIcon = function (col) { return state.sortCol === col ? (state.sortAsc ? ' &#8593;' : ' &#8595;') : ''; };

    var trs = rows.map(function (t) {
      return '<tr>' +
        '<td>' + esc(t.date) + '</td>' +
        '<td><span class="acct-badge ' + (t.type === 'income' ? 'acct-badge-green' : 'acct-badge-red') + '">' + t.type + '</span></td>' +
        '<td><span class="acct-cat-pill">' + esc(t.category) + '</span></td>' +
        '<td>' + esc(t.description) + '</td>' +
        '<td class="acct-amount ' + t.type + '">' + (t.type === 'income' ? '+' : '-') + '$' + fmt(t.amount) + '</td>' +
        '<td style="white-space:nowrap">' +
          '<button class="acct-row-btn acct-edit-btn" data-id="' + t.id + '" title="Edit">&#9998;</button>' +
          '<button class="acct-row-btn acct-del-btn" data-id="' + t.id + '" title="Delete">&#10005;</button>' +
        '</td>' +
      '</tr>';
    }).join('');

    return '<div class="acct-table-wrap"><table class="acct-table">' +
      '<thead><tr>' +
        '<th class="acct-sortable" data-col="date">Date' + sortIcon('date') + '</th>' +
        '<th>Type</th>' +
        '<th class="acct-sortable" data-col="category">Category' + sortIcon('category') + '</th>' +
        '<th class="acct-sortable" data-col="description">Description' + sortIcon('description') + '</th>' +
        '<th class="acct-sortable" data-col="amount">Amount' + sortIcon('amount') + '</th>' +
        '<th></th>' +
      '</tr></thead>' +
      '<tbody>' + trs + '</tbody>' +
    '</table></div>';
  }

  function showForm(txn) {
    var area = document.getElementById('acct-form-area');
    if (!area) return;
    state.editingId = txn ? txn.id : null;
    var isEdit = !!txn;

    var catOpts = state.categories.map(function (c) {
      return '<option value="' + esc(c) + '"' + (txn && txn.category === c ? ' selected' : '') + '>' + esc(c) + '</option>';
    }).join('');

    area.innerHTML =
      '<div class="acct-form">' +
        '<div class="acct-form-title">' + (isEdit ? 'Edit' : 'New') + ' Transaction</div>' +
        '<div class="acct-form-grid">' +
          '<div class="acct-form-field"><label>Date</label><input type="date" id="acct-f-date" value="' + (txn ? txn.date : new Date().toISOString().slice(0, 10)) + '"></div>' +
          '<div class="acct-form-field"><label>Type</label><select id="acct-f-type"><option value="income"' + (txn && txn.type === 'income' ? ' selected' : '') + '>Income</option><option value="expense"' + (txn && txn.type === 'expense' ? ' selected' : '') + '>Expense</option></select></div>' +
          '<div class="acct-form-field"><label>Category</label><select id="acct-f-category">' + catOpts + '</select></div>' +
          '<div class="acct-form-field"><label>Amount ($)</label><input type="number" id="acct-f-amount" step="0.01" min="0" placeholder="0.00" value="' + (txn ? txn.amount : '') + '"></div>' +
          '<div class="acct-form-field acct-form-field-wide"><label>Description</label><input type="text" id="acct-f-desc" placeholder="Transaction description" value="' + (txn ? esc(txn.description) : '') + '"></div>' +
        '</div>' +
        '<div class="acct-form-actions">' +
          '<button class="acct-btn acct-btn-primary" id="acct-f-save">' + (isEdit ? 'Update' : 'Save') + '</button>' +
          '<button class="acct-btn acct-btn-secondary" id="acct-f-cancel">Cancel</button>' +
        '</div>' +
      '</div>';

    document.getElementById('acct-f-save').addEventListener('click', saveTransaction);
    document.getElementById('acct-f-cancel').addEventListener('click', function () { area.innerHTML = ''; state.editingId = null; });
    var descInput = document.getElementById('acct-f-desc');
    if (descInput) descInput.focus();
  }

  function saveTransaction() {
    var date = (document.getElementById('acct-f-date') || {}).value;
    var type = (document.getElementById('acct-f-type') || {}).value;
    var category = (document.getElementById('acct-f-category') || {}).value;
    var amount = parseFloat((document.getElementById('acct-f-amount') || {}).value);
    var description = ((document.getElementById('acct-f-desc') || {}).value || '').trim();

    if (!date || !description || isNaN(amount) || amount <= 0) {
      toast('Please fill all fields with valid values');
      return;
    }

    if (state.editingId) {
      for (var i = 0; i < state.transactions.length; i++) {
        if (state.transactions[i].id === state.editingId) {
          state.transactions[i] = { id: state.editingId, date: date, type: type, category: category, amount: amount, description: description };
          break;
        }
      }
      toast('Transaction updated');
    } else {
      state.transactions.unshift({ id: uid(), date: date, type: type, category: category, amount: amount, description: description });
      toast('Transaction added');
    }

    saveData();
    state.editingId = null;
    render();
  }

  function exportCSV() {
    if (state.transactions.length === 0) { toast('No transactions to export'); return; }
    var header = 'Date,Type,Category,Description,Amount';
    var rows = state.transactions.map(function (t) {
      return t.date + ',' + t.type + ',"' + t.category + '","' + t.description.replace(/"/g, '""') + '",' + (t.type === 'income' ? '' : '-') + t.amount.toFixed(2);
    });
    var csv = [header].concat(rows).join('\n');
    var blob = new Blob([csv], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'sv-accounting-' + new Date().toISOString().slice(0, 10) + '.csv';
    a.click();
    URL.revokeObjectURL(url);
    toast('CSV exported');
  }

  function bindEvents() {
    var addBtn = document.getElementById('acct-add-btn');
    if (addBtn) addBtn.addEventListener('click', function () { showForm(); });

    var exportBtn = document.getElementById('acct-export-btn');
    if (exportBtn) exportBtn.addEventListener('click', exportCSV);

    var page = document.getElementById(PAGE_ID);
    if (!page) return;

    // Filters
    page.querySelectorAll('.acct-filter-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.filter = btn.dataset.filter;
        render();
      });
    });

    // Search
    var search = document.getElementById('acct-search');
    if (search) {
      search.addEventListener('input', function () {
        state.search = search.value;
        var tableArea = document.getElementById('acct-table-area');
        if (tableArea) tableArea.innerHTML = renderTable();
        bindTableEvents();
      });
    }

    bindTableEvents();
  }

  function bindTableEvents() {
    var page = document.getElementById(PAGE_ID);
    if (!page) return;

    // Sortable headers
    page.querySelectorAll('.acct-sortable').forEach(function (th) {
      th.addEventListener('click', function () {
        var col = th.dataset.col;
        if (state.sortCol === col) { state.sortAsc = !state.sortAsc; }
        else { state.sortCol = col; state.sortAsc = true; }
        render();
      });
    });

    // Edit
    page.querySelectorAll('.acct-edit-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var txn = null;
        for (var i = 0; i < state.transactions.length; i++) {
          if (state.transactions[i].id === btn.dataset.id) { txn = state.transactions[i]; break; }
        }
        if (txn) showForm(txn);
      });
    });

    // Delete
    page.querySelectorAll('.acct-del-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.transactions = state.transactions.filter(function (t) { return t.id !== btn.dataset.id; });
        saveData();
        render();
        toast('Transaction deleted');
      });
    });
  }

  // ── Init ──

  function shouldRender() {
    return window.location.pathname === '/accounting' || window.location.pathname.indexOf('/accounting') === 0;
  }

  function init() {
    if (!shouldRender()) {
      // Clean up if navigated away
      var existing = document.getElementById(PAGE_ID);
      if (existing) existing.remove();
      return;
    }

    injectStyles();
    loadData();

    if (!document.getElementById(PAGE_ID)) {
      var page = document.createElement('div');
      page.id = PAGE_ID;
      document.body.appendChild(page);
    }

    render();
  }

  // Run on load and watch for navigation changes
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { setTimeout(init, 400); });
  } else {
    setTimeout(init, 400);
  }

  // Re-check periodically for SPA navigation
  setInterval(function () {
    var exists = document.getElementById(PAGE_ID);
    if (shouldRender() && !exists) {
      init();
    } else if (!shouldRender() && exists) {
      exists.remove();
    }
  }, 1000);

})();
