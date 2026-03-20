/**
 * Nodes page — view connected device nodes, approve pairing, invoke commands.
 *
 * Uses WS frames: node.list, node.pair.approve, node.invoke,
 * and listens for: node.connected, node.disconnected, node.pair.pending
 */

window.NodesPage = {
  _nodes: [],
  _pendingPairings: [],
  _unsubs: [],

  render(container) {
    container.innerHTML = `
      <div class="page">
        <div class="page-header">
          <span class="page-title">Nodes</span>
          <button class="btn btn-primary" id="nodes-refresh">refresh</button>
        </div>
        <div class="page-body">
          <div id="nodes-pending"></div>
          <div id="nodes-content"></div>
        </div>
      </div>
    `;

    document.getElementById('nodes-refresh').addEventListener('click', () => this._load());

    // Listen for real-time node events
    const unConn = window.nanoWS.on('node.connected', () => this._load());
    const unDisc = window.nanoWS.on('node.disconnected', () => this._load());
    const unPair = window.nanoWS.on('node.pair.pending', (frame) => {
      this._pendingPairings.push(frame.node || frame);
      this._renderPending();
    });
    this._unsubs.push(unConn, unDisc, unPair);

    this._load();
  },

  destroy() {
    this._unsubs.forEach(fn => fn());
    this._unsubs = [];
  },

  async _load() {
    const content = document.getElementById('nodes-content');
    if (!content) return;

    if (!window.nanoWS.connected) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📱</div>
          <div class="empty-state-text">Connect to the gateway to view nodes.</div>
        </div>
      `;
      return;
    }

    try {
      const resp = await window.nanoWS.request('node.list');
      this._nodes = resp.nodes || [];
      this._renderNodes(content);
    } catch (e) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⚠</div>
          <div class="empty-state-text">${this._escapeHtml(e.message)}</div>
        </div>
      `;
    }
  },

  _renderPending() {
    const container = document.getElementById('nodes-pending');
    if (!container) return;

    if (this._pendingPairings.length === 0) {
      container.innerHTML = '';
      return;
    }

    container.innerHTML = `
      <div class="config-section">
        <div class="config-section-title" style="color:var(--yellow)">⏳ Pending Pairing Requests</div>
        ${this._pendingPairings.map((node, i) => `
          <div class="node-card" style="border-color:var(--yellow)">
            <div class="node-card-header">
              <div class="node-name">${this._escapeHtml(node.name || node.device_id)}</div>
              <div style="display:flex;gap:6px;">
                <button class="btn btn-primary" data-pair-idx="${i}" data-approve="true">approve</button>
                <button class="btn" data-pair-idx="${i}" data-approve="false" style="color:var(--red);border-color:var(--red)">reject</button>
              </div>
            </div>
            <div class="node-meta">
              <span class="badge badge-yellow">${this._escapeHtml(node.platform || 'unknown')}</span>
              <span style="color:var(--text-muted);font-size:11px;">ID: ${this._escapeHtml(node.device_id)}</span>
            </div>
            ${(node.capabilities || []).length > 0 ? `
              <div class="node-capabilities">
                ${node.capabilities.map(c => `<span class="badge badge-blue">${this._escapeHtml(c)}</span>`).join(' ')}
              </div>
            ` : ''}
          </div>
        `).join('')}
      </div>
    `;

    // Bind approve/reject buttons
    container.querySelectorAll('[data-pair-idx]').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.pairIdx);
        const approve = btn.dataset.approve === 'true';
        this._handlePairing(idx, approve);
      });
    });
  },

  async _handlePairing(idx, approve) {
    const node = this._pendingPairings[idx];
    if (!node) return;

    try {
      await window.nanoWS.request('node.pair.approve', {
        device_id: node.device_id,
        approve,
      });
      this._pendingPairings.splice(idx, 1);
      this._renderPending();
      window.app?.toast(`Node ${approve ? 'approved' : 'rejected'}`, approve ? 'success' : 'info');
      if (approve) this._load(); // Refresh node list
    } catch (e) {
      window.app?.toast(`Pairing failed: ${e.message}`, 'error');
    }
  },

  _renderNodes(container) {
    if (this._nodes.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">◇</div>
          <div class="empty-state-text">No connected nodes.<br><span style="font-size:11px;color:var(--text-muted)">Nodes connect via WS with role: "node"</span></div>
        </div>
      `;
      return;
    }

    container.innerHTML = `
      <div class="node-grid">
        ${this._nodes.map(node => this._renderNodeCard(node)).join('')}
      </div>
    `;

    // Bind invoke buttons
    container.querySelectorAll('[data-invoke]').forEach(btn => {
      btn.addEventListener('click', () => {
        this._invokeCommand(btn.dataset.deviceId, btn.dataset.invoke);
      });
    });
  },

  _renderNodeCard(node) {
    const name = node.name || node.device_id || 'Unknown';
    const platform = node.platform || 'unknown';
    const capabilities = node.capabilities || [];
    const online = node.online !== false;
    const battery = node.battery;
    const deviceId = node.device_id || '';

    let batteryStr = '';
    if (battery != null) {
      const pct = Math.round(battery * 100);
      const color = pct > 50 ? 'var(--green)' : pct > 20 ? 'var(--yellow)' : 'var(--red)';
      batteryStr = `<span style="color:${color};font-size:11px;">⚡ ${pct}%</span>`;
    }

    const platformIcon = {
      ios: '📱', android: '📱', macos: '💻', linux: '🖥', windows: '🖥'
    }[platform] || '📟';

    // Quick-invoke buttons for common commands
    const invokeButtons = capabilities.map(cap => {
      const cmds = {
        camera: 'camera.capture',
        screen: 'screen.capture',
        location: 'location.get',
        clipboard: 'clipboard.read',
      };
      const cmd = cmds[cap];
      if (!cmd) return '';
      return `<button class="btn" data-invoke="${cmd}" data-device-id="${this._escapeAttr(deviceId)}" style="font-size:10px;padding:2px 6px;">${cmd}</button>`;
    }).filter(Boolean).join(' ');

    return `
      <div class="node-card">
        <div class="node-card-header">
          <div class="node-name">${platformIcon} ${this._escapeHtml(name)}</div>
          <span class="badge ${online ? 'badge-green' : 'badge-red'}">${online ? 'online' : 'offline'}</span>
        </div>
        <div class="node-meta">
          <span class="badge badge-blue">${this._escapeHtml(platform)}</span>
          ${batteryStr}
          <span style="color:var(--text-muted);font-size:10px;">${this._escapeHtml(deviceId).substring(0, 16)}</span>
        </div>
        ${capabilities.length > 0 ? `
          <div class="node-capabilities">
            ${capabilities.map(c => `<span class="badge badge-blue">${this._escapeHtml(c)}</span>`).join(' ')}
          </div>
        ` : ''}
        ${invokeButtons ? `<div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap;">${invokeButtons}</div>` : ''}
      </div>
    `;
  },

  async _invokeCommand(deviceId, command) {
    try {
      window.app?.toast(`Invoking ${command}...`, 'info');
      const resp = await window.nanoWS.request('node.invoke', {
        device_id: deviceId,
        command,
        params: {},
      }, 35000);
      if (resp.success === false) {
        window.app?.toast(`Command failed: ${resp.error || 'unknown'}`, 'error');
      } else {
        window.app?.toast(`${command} completed`, 'success');
      }
    } catch (e) {
      window.app?.toast(`Invoke failed: ${e.message}`, 'error');
    }
  },

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  _escapeAttr(text) {
    return text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  },
};
