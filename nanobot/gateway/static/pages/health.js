/**
 * Health page — system health monitoring, service status, and log viewer.
 *
 * Shows: connection status, gateway info, service health, recent logs.
 */

window.HealthPage = {
  _health: null,
  _logs: [],
  _unsubs: [],
  _refreshInterval: null,

  render(container) {
    container.innerHTML = `
      <div class="page">
        <div class="page-header">
          <span class="page-title">Health</span>
          <button class="btn btn-primary" id="health-refresh">refresh</button>
        </div>
        <div class="page-body">
          <div id="health-grid"></div>
          <div id="health-logs"></div>
        </div>
      </div>
    `;

    document.getElementById('health-refresh').addEventListener('click', () => this._load());

    // Auto-refresh every 10s
    this._refreshInterval = setInterval(() => this._load(), 10000);

    this._load();
  },

  destroy() {
    this._unsubs.forEach(fn => fn());
    this._unsubs = [];
    if (this._refreshInterval) {
      clearInterval(this._refreshInterval);
      this._refreshInterval = null;
    }
  },

  async _load() {
    const grid = document.getElementById('health-grid');
    if (!grid) return;

    if (!window.nanoWS.connected) {
      grid.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">💓</div>
          <div class="empty-state-text">Connect to the gateway to view health status.</div>
        </div>
      `;
      document.getElementById('health-logs').innerHTML = '';
      return;
    }

    // Gather health data from multiple sources
    const health = {
      gateway: { status: 'ok', label: 'Gateway' },
      websocket: { status: 'ok', label: 'WebSocket' },
      sessions: { status: 'unknown', label: 'Sessions', detail: '—' },
      nodes: { status: 'unknown', label: 'Nodes', detail: '—' },
      config: { status: 'unknown', label: 'Config', detail: '—' },
    };

    // Check sessions
    try {
      const sessResp = await window.nanoWS.request('session.list');
      const count = (sessResp.sessions || []).length;
      health.sessions.status = 'ok';
      health.sessions.detail = `${count} active`;
    } catch {
      health.sessions.status = 'error';
      health.sessions.detail = 'unavailable';
    }

    // Check nodes
    try {
      const nodeResp = await window.nanoWS.request('node.list');
      const nodes = nodeResp.nodes || [];
      const online = nodes.filter(n => n.online !== false).length;
      health.nodes.status = nodes.length > 0 ? 'ok' : 'warn';
      health.nodes.detail = `${online}/${nodes.length} online`;
    } catch {
      health.nodes.status = 'warn';
      health.nodes.detail = 'no nodes';
    }

    // Check config
    try {
      const cfgResp = await window.nanoWS.request('config.get', { path: '' });
      const cfg = cfgResp.config || cfgResp.value || {};
      const model = cfg.agents?.defaults?.model || 'not set';
      health.config.status = model !== 'not set' ? 'ok' : 'warn';
      health.config.detail = model;
    } catch {
      health.config.status = 'error';
      health.config.detail = 'unavailable';
    }

    // Ping latency
    try {
      const t0 = performance.now();
      await window.nanoWS.request('ping');
      const latency = Math.round(performance.now() - t0);
      health.websocket.detail = `${latency}ms latency`;
    } catch {
      health.websocket.detail = 'ping failed';
      health.websocket.status = 'warn';
    }

    health.gateway.detail = window.nanoWS._url || 'connected';

    this._health = health;
    this._renderGrid(grid);
    this._renderLogs();
  },

  _renderGrid(container) {
    const h = this._health;
    if (!h) return;

    const cards = Object.values(h).map(svc => {
      const statusClass = {
        ok: 'health-ok',
        warn: 'health-warn',
        error: 'health-error',
        unknown: 'health-unknown',
      }[svc.status] || 'health-unknown';

      const statusIcon = {
        ok: '●',
        warn: '◐',
        error: '✕',
        unknown: '○',
      }[svc.status] || '○';

      return `
        <div class="health-card ${statusClass}">
          <div class="health-card-header">
            <span class="health-status-icon">${statusIcon}</span>
            <span class="health-card-title">${svc.label}</span>
          </div>
          <div class="health-card-detail">${this._escapeHtml(svc.detail || '')}</div>
        </div>
      `;
    }).join('');

    container.innerHTML = `<div class="health-grid">${cards}</div>`;
  },

  _renderLogs() {
    const container = document.getElementById('health-logs');
    if (!container) return;

    // Show gateway capabilities
    const methods = window.nanoWS.methods || [];
    const events = window.nanoWS.events || [];

    container.innerHTML = `
      <div class="config-section" style="margin-top:20px;">
        <div class="config-section-title">Gateway Capabilities</div>
        <div class="config-row">
          <div class="config-key">Methods</div>
          <div class="config-readonly">
            ${methods.length > 0
              ? methods.map(m => `<span class="badge badge-blue">${this._escapeHtml(m)}</span>`).join(' ')
              : '<span style="color:var(--text-muted)">none</span>'
            }
          </div>
        </div>
        <div class="config-row">
          <div class="config-key">Events</div>
          <div class="config-readonly">
            ${events.length > 0
              ? events.map(e => `<span class="badge badge-green">${this._escapeHtml(e)}</span>`).join(' ')
              : '<span style="color:var(--text-muted)">none</span>'
            }
          </div>
        </div>
      </div>

      <div class="config-section" style="margin-top:20px;">
        <div class="config-section-title">Connection Info</div>
        <div class="config-row">
          <div class="config-key">URL</div>
          <div class="config-readonly">${this._escapeHtml(window.nanoWS._url || '—')}</div>
        </div>
        <div class="config-row">
          <div class="config-key">Status</div>
          <div class="config-readonly">
            <span class="badge badge-green">${window.nanoWS.connected ? 'connected' : 'disconnected'}</span>
          </div>
        </div>
        <div class="config-row">
          <div class="config-key">Protocol</div>
          <div class="config-readonly">v1</div>
        </div>
      </div>
    `;
  },

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },
};
