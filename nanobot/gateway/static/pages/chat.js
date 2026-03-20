/**
 * Chat page — real-time conversation with agent via WS gateway.
 *
 * Supports streaming token-by-token responses, session selection,
 * and chat history loading.
 */

window.ChatPage = {
  _sessionKey: 'dashboard:chat:default',
  _messages: [],
  _streaming: false,
  _streamBuffer: '',
  _unsubs: [],

  render(container) {
    container.innerHTML = `
      <div class="chat-container">
        <div class="page-header">
          <span class="page-title">Chat</span>
          <div class="session-select">
            <label style="color:var(--text-muted);font-size:11px;">session:</label>
            <select id="chat-session-select">
              <option value="dashboard:chat:default">default</option>
            </select>
            <button class="btn btn-primary" id="chat-new-session">+ new</button>
          </div>
        </div>
        <div class="chat-messages" id="chat-messages"></div>
        <div class="chat-input-area">
          <textarea id="chat-input" placeholder="Send a message…" rows="1"></textarea>
          <button class="chat-send-btn" id="chat-send">send</button>
        </div>
      </div>
    `;

    this._bindEvents();
    this._loadSessions();
  },

  destroy() {
    this._unsubs.forEach(fn => fn());
    this._unsubs = [];
  },

  _bindEvents() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const newBtn = document.getElementById('chat-new-session');
    const sessionSelect = document.getElementById('chat-session-select');

    // Send on click
    sendBtn.addEventListener('click', () => this._send());

    // Send on Enter (Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._send();
      }
    });

    // Auto-resize textarea
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    });

    // New session
    newBtn.addEventListener('click', () => {
      const ts = Date.now();
      this._sessionKey = `dashboard:chat:${ts}`;
      this._messages = [];
      this._renderMessages();
      const opt = document.createElement('option');
      opt.value = this._sessionKey;
      opt.textContent = new Date(ts).toLocaleTimeString();
      sessionSelect.appendChild(opt);
      sessionSelect.value = this._sessionKey;
    });

    // Switch session
    sessionSelect.addEventListener('change', () => {
      this._sessionKey = sessionSelect.value;
      this._messages = [];
      this._renderMessages();
    });

    // Listen for streaming tokens
    const unToken = window.nanoWS.on('chat.token', (frame) => {
      if (this._streaming) {
        this._streamBuffer += (frame.token || '');
        this._updateStreamingMessage();
      }
    });
    this._unsubs.push(unToken);

    // Listen for complete responses
    const unResp = window.nanoWS.on('chat.response', (frame) => {
      this._streaming = false;
      const lastMsg = this._messages[this._messages.length - 1];
      if (lastMsg && lastMsg.role === 'assistant' && lastMsg.streaming) {
        lastMsg.content = frame.content || this._streamBuffer;
        lastMsg.streaming = false;
      }
      this._streamBuffer = '';
      this._renderMessages();
      this._setInputEnabled(true);
    });
    this._unsubs.push(unResp);

    // Listen for errors
    const unErr = window.nanoWS.on('error', (frame) => {
      if (this._streaming) {
        this._streaming = false;
        this._streamBuffer = '';
        this._messages.push({
          role: 'system',
          content: `Error: ${frame.message || 'Unknown error'}`,
        });
        this._renderMessages();
        this._setInputEnabled(true);
      }
    });
    this._unsubs.push(unErr);
  },

  async _send() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text || this._streaming) return;

    input.value = '';
    input.style.height = 'auto';

    // Add user message
    this._messages.push({ role: 'user', content: text });

    // Add placeholder for streaming response
    this._messages.push({ role: 'assistant', content: '', streaming: true });
    this._streaming = true;
    this._streamBuffer = '';
    this._renderMessages();
    this._setInputEnabled(false);

    // Send via WS
    try {
      window.nanoWS.send('chat.send', {
        content: text,
        session_key: this._sessionKey,
        channel: 'dashboard',
        chat_id: 'dashboard',
      });
    } catch (e) {
      this._streaming = false;
      this._messages.pop(); // Remove placeholder
      this._messages.push({ role: 'system', content: `Send failed: ${e.message}` });
      this._renderMessages();
      this._setInputEnabled(true);
    }
  },

  _updateStreamingMessage() {
    const lastMsg = this._messages[this._messages.length - 1];
    if (lastMsg && lastMsg.streaming) {
      lastMsg.content = this._streamBuffer;
    }
    // Update only the last message element for performance
    const msgs = document.getElementById('chat-messages');
    const lastEl = msgs?.lastElementChild;
    if (lastEl && lastEl.classList.contains('chat-streaming')) {
      lastEl.textContent = this._streamBuffer;
    }
    this._scrollToBottom();
  },

  _renderMessages() {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    if (this._messages.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">⬡</div>
          <div class="empty-state-text">Send a message to start chatting with the agent.</div>
        </div>
      `;
      return;
    }

    container.innerHTML = this._messages.map(m => {
      const cls = m.streaming ? 'chat-msg assistant chat-streaming' : `chat-msg ${m.role}`;
      return `<div class="${cls}">${this._escapeHtml(m.content)}</div>`;
    }).join('');

    this._scrollToBottom();
  },

  _scrollToBottom() {
    const container = document.getElementById('chat-messages');
    if (container) container.scrollTop = container.scrollHeight;
  },

  _setInputEnabled(enabled) {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('chat-send');
    if (input) input.disabled = !enabled;
    if (btn) btn.disabled = !enabled;
    if (enabled && input) input.focus();
  },

  async _loadSessions() {
    if (!window.nanoWS.connected) return;
    try {
      const resp = await window.nanoWS.request('session.list');
      const sessions = resp.sessions || [];
      const select = document.getElementById('chat-session-select');
      if (!select) return;
      // Add sessions from server
      sessions.forEach(key => {
        if (key !== 'dashboard:chat:default') {
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = key.replace('dashboard:chat:', '').substring(0, 20);
          select.appendChild(opt);
        }
      });
    } catch {
      // Sessions not available yet, that's fine
    }
  },

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },
};
