/**
 * nanobot dashboard — SPA controller.
 *
 * Handles hash-based routing, page lifecycle (render/destroy),
 * WS connection management, toast notifications, and status indicator.
 */

window.app = {
  _currentPage: null,
  _currentPageName: null,
  _toastTimer: null,

  /** Bootstrap the app. Called once from index.html. */
  init() {
    this._bindNav();
    this._bindConnection();
    this._bindStatus();

    // Route to current hash or default
    window.addEventListener('hashchange', () => this._route());
    this._route();

    // Auto-connect if on same host
    this._autoConnect();
  },

  // ── Routing ──

  _route() {
    const hash = location.hash.replace(/^#\/?/, '') || 'chat';
    const pages = {
      chat: window.ChatPage,
      sessions: window.SessionsPage,
      config: window.ConfigPage,
      nodes: window.NodesPage,
      health: window.HealthPage,
      accounting: window.AccountingPage,
    };

    const page = pages[hash];
    if (!page) {
      location.hash = '#chat';
      return;
    }

    // Destroy previous page
    if (this._currentPage && this._currentPage.destroy) {
      this._currentPage.destroy();
    }

    // Update active nav
    document.querySelectorAll('.nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.page === hash);
    });

    // Render new page into the content area (not the whole app shell)
    const container = document.getElementById('content');
    if (container) {
      this._currentPage = page;
      this._currentPageName = hash;
      page.render(container);
    }
  },

  _bindNav() {
    document.querySelectorAll('.nav-item').forEach(el => {
      el.addEventListener('click', () => {
        location.hash = '#' + el.dataset.page;
      });
    });
  },

  // ── Connection ──

  _bindConnection() {
    const connectBtn = document.getElementById('connect-btn');
    const urlInput = document.getElementById('ws-url');
    const tokenInput = document.getElementById('ws-token');

    connectBtn?.addEventListener('click', () => {
      if (window.nanoWS.connected) {
        window.nanoWS.disconnect();
      } else {
        const url = urlInput?.value || `ws://${location.host}/ws`;
        const token = tokenInput?.value || '';
        window.nanoWS.connect(url, token);
      }
    });
  },

  _autoConnect() {
    // If served from the gateway itself, auto-connect
    if (location.protocol === 'http:' || location.protocol === 'https:') {
      const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${wsProto}//${location.host}/ws`;
      const urlInput = document.getElementById('ws-url');
      if (urlInput) urlInput.value = url;
      window.nanoWS.connect(url);
    }
  },

  // ── Status indicator ──

  _bindStatus() {
    const connStatus = document.getElementById('conn-status');
    const indicator = connStatus?.querySelector('.status-dot');
    const statusText = connStatus?.querySelector('.status-text');

    window.nanoWS.on('status', (status) => {
      if (connStatus) {
        connStatus.className = 'status-indicator ' + status;
      }
      if (statusText) {
        statusText.textContent = status;
      }

      // Refresh current page on reconnect
      if (status === 'connected' && this._currentPage) {
        const container = document.getElementById('content');
        if (container && this._currentPage.render) {
          this._currentPage.render(container);
        }
      }
    });

    window.nanoWS.on('hello', (frame) => {
      this.toast('Connected to gateway', 'success');
    });

    window.nanoWS.on('error', (frame) => {
      if (frame.message && !frame.id) {
        this.toast(`Error: ${frame.message}`, 'error');
      }
    });
  },

  // ── Toast notifications ──

  toast(message, type = 'info') {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    if (this._toastTimer) clearTimeout(this._toastTimer);

    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    document.body.appendChild(el);

    // Trigger animation
    requestAnimationFrame(() => el.classList.add('toast-visible'));

    this._toastTimer = setTimeout(() => {
      el.classList.remove('toast-visible');
      setTimeout(() => el.remove(), 300);
    }, 3000);
  },
};

// Boot
document.addEventListener('DOMContentLoaded', () => window.app.init());
