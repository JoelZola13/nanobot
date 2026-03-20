/**
 * Config page — view and edit runtime configuration via WS gateway.
 *
 * Read-only fields show current config; editable fields are the
 * runtime-mutable paths defined in gateway/server.py:
 *   agents.defaults.model, temperature, max_tokens, max_tool_iterations, memory_window
 */

window.ConfigPage = {
  _config: null,
  _unsubs: [],

  render(container) {
    container.innerHTML = `
      <div class="page">
        <div class="page-header">
          <span class="page-title">Configuration</span>
          <button class="btn btn-primary" id="config-refresh">refresh</button>
        </div>
        <div class="page-body">
          <div id="config-content"></div>
        </div>
      </div>
    `;

    document.getElementById('config-refresh').addEventListener('click', () => this._load());
    this._load();
  },

  destroy() {
    this._unsubs.forEach(fn => fn());
    this._unsubs = [];
  },

  async _load() {
    const content = document.getElementById('config-content');
    if (!content) return;

    if (!window.nanoWS.connected) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⚙</div>
          <div class="empty-state-text">Connect to the gateway to view configuration.</div>
        </div>
      `;
      return;
    }

    try {
      const resp = await window.nanoWS.request('config.get', { path: '' });
      this._config = resp.config || resp.value || {};
      this._renderConfig(content);
    } catch (e) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⚠</div>
          <div class="empty-state-text">${e.message}</div>
        </div>
      `;
    }
  },

  _renderConfig(container) {
    const agents = this._config.agents?.defaults || {};
    const gateway = this._config.gateway || {};
    const channels = this._config.channels || {};
    const tools = this._config.tools || {};

    // Editable fields
    const editableFields = [
      { path: 'agents.defaults.model', label: 'Model', value: agents.model || '', type: 'text' },
      { path: 'agents.defaults.temperature', label: 'Temperature', value: agents.temperature ?? 0.7, type: 'number', step: '0.1', min: '0', max: '2' },
      { path: 'agents.defaults.max_tokens', label: 'Max Tokens', value: agents.maxTokens ?? agents.max_tokens ?? 8192, type: 'number' },
      { path: 'agents.defaults.max_tool_iterations', label: 'Max Tool Iterations', value: agents.maxToolIterations ?? agents.max_tool_iterations ?? 20, type: 'number' },
      { path: 'agents.defaults.memory_window', label: 'Memory Window', value: agents.memoryWindow ?? agents.memory_window ?? 50, type: 'number' },
    ];

    // Enabled channels
    const enabledChannels = Object.entries(channels)
      .filter(([_, v]) => v && v.enabled)
      .map(([name]) => name);

    // MCP servers
    const mcpServers = Object.keys(tools.mcpServers || tools.mcp_servers || {});

    let html = `
      <div class="config-section">
        <div class="config-section-title">Agent Defaults (editable)</div>
        ${editableFields.map(f => `
          <div class="config-row">
            <div class="config-key">${f.label}</div>
            <div class="config-value">
              <input type="${f.type}" id="cfg-${f.path}" value="${this._escapeAttr(String(f.value))}"
                ${f.step ? `step="${f.step}"` : ''} ${f.min ? `min="${f.min}"` : ''} ${f.max ? `max="${f.max}"` : ''}>
            </div>
            <button class="config-save-btn" data-path="${f.path}" data-input="cfg-${f.path}">save</button>
          </div>
        `).join('')}
      </div>

      <div class="config-section">
        <div class="config-section-title">Gateway</div>
        <div class="config-row">
          <div class="config-key">Host</div>
          <div class="config-readonly">${gateway.host || '0.0.0.0'}</div>
        </div>
        <div class="config-row">
          <div class="config-key">Port</div>
          <div class="config-readonly">${gateway.port || 18790}</div>
        </div>
        <div class="config-row">
          <div class="config-key">Auth tokens</div>
          <div class="config-readonly">${(gateway.authTokens || gateway.auth_tokens || []).length} configured</div>
        </div>
      </div>

      <div class="config-section">
        <div class="config-section-title">Channels</div>
        <div class="config-row">
          <div class="config-key">Enabled</div>
          <div class="config-readonly">
            ${enabledChannels.length > 0
              ? enabledChannels.map(c => `<span class="badge badge-green">${c}</span>`).join(' ')
              : '<span class="badge badge-yellow">none</span>'
            }
          </div>
        </div>
      </div>

      <div class="config-section">
        <div class="config-section-title">Tools</div>
        <div class="config-row">
          <div class="config-key">MCP servers</div>
          <div class="config-readonly">
            ${mcpServers.length > 0
              ? mcpServers.map(s => `<span class="badge badge-blue">${s}</span>`).join(' ')
              : '<span style="color:var(--text-muted)">none</span>'
            }
          </div>
        </div>
        <div class="config-row">
          <div class="config-key">Restrict to workspace</div>
          <div class="config-readonly">${tools.restrictToWorkspace || tools.restrict_to_workspace ? 'yes' : 'no'}</div>
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Bind save buttons
    container.querySelectorAll('.config-save-btn').forEach(btn => {
      btn.addEventListener('click', () => this._save(btn.dataset.path, btn.dataset.input));
    });
  },

  async _save(path, inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;

    let value = input.value;
    // Coerce numbers
    if (input.type === 'number') {
      value = Number(value);
      if (isNaN(value)) {
        window.app?.toast('Invalid number', 'error');
        return;
      }
    }

    try {
      await window.nanoWS.request('config.set', { path, value });
      window.app?.toast(`Saved ${path}`, 'success');
    } catch (e) {
      window.app?.toast(`Failed: ${e.message}`, 'error');
    }
  },

  _escapeAttr(text) {
    return text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  },
};
