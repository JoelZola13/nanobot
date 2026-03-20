/**
 * Sessions page — view and manage active agent sessions.
 */

window.SessionsPage = {
  _sessions: [],
  _unsubs: [],

  render(container) {
    container.innerHTML = `
      <div class="page">
        <div class="page-header">
          <span class="page-title">Sessions</span>
          <button class="btn btn-primary" id="sessions-refresh">refresh</button>
        </div>
        <div class="page-body">
          <div id="sessions-content"></div>
        </div>
      </div>
    `;

    document.getElementById('sessions-refresh').addEventListener('click', () => this._load());
    this._load();
  },

  destroy() {
    this._unsubs.forEach(fn => fn());
    this._unsubs = [];
  },

  async _load() {
    const content = document.getElementById('sessions-content');
    if (!content) return;

    if (!window.nanoWS.connected) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⚡</div>
          <div class="empty-state-text">Connect to the gateway to view sessions.</div>
        </div>
      `;
      return;
    }

    try {
      const resp = await window.nanoWS.request('session.list');
      this._sessions = resp.sessions || [];
      this._renderTable(content);
    } catch (e) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⚠</div>
          <div class="empty-state-text">${e.message}</div>
        </div>
      `;
    }
  },

  _renderTable(container) {
    if (this._sessions.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">◇</div>
          <div class="empty-state-text">No active sessions.</div>
        </div>
      `;
      return;
    }

    const rows = this._sessions.map((key, i) => {
      // Parse session key to extract channel and info
      const parts = key.split(':');
      const channel = parts[0] || '—';
      const chatId = parts.slice(1).join(':') || '—';

      return `
        <tr>
          <td>${i + 1}</td>
          <td style="color:var(--cyan)">${this._escapeHtml(key)}</td>
          <td><span class="badge badge-blue">${this._escapeHtml(channel)}</span></td>
          <td>${this._escapeHtml(chatId)}</td>
          <td><span class="badge badge-green">active</span></td>
        </tr>
      `;
    }).join('');

    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Session Key</th>
            <th>Channel</th>
            <th>Chat ID</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  },

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },
};
