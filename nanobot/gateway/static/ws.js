/**
 * nanobot WS gateway client.
 *
 * Handles connection, auth, frame routing, and reconnection.
 * All frames follow the schema in nanobot/gateway/protocol.py.
 */

class NanobotWS {
  constructor() {
    this._ws = null;
    this._url = null;
    this._token = '';
    this._requestId = 0;
    this._pending = new Map(); // id -> {resolve, reject, timeout}
    this._listeners = new Map(); // event -> Set<callback>
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._reconnectTimer = null;
    this._connected = false;
    this._methods = [];
    this._events = [];
    this._intentionalClose = false;
  }

  /** Connect to the gateway WS endpoint. */
  connect(url, token) {
    this._url = url || `ws://${location.host}/ws`;
    this._token = token || '';
    this._intentionalClose = false;
    this._doConnect();
  }

  disconnect() {
    this._intentionalClose = true;
    if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    if (this._ws) this._ws.close(1000, 'client disconnect');
  }

  get connected() { return this._connected; }
  get methods() { return this._methods; }
  get events() { return this._events; }

  /** Register an event listener. Returns unsubscribe function. */
  on(event, callback) {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event).add(callback);
    return () => this._listeners.get(event)?.delete(callback);
  }

  /** Send a request frame and wait for correlated response. */
  async request(type, payload = {}, timeoutMs = 15000) {
    if (!this._connected) throw new Error('Not connected');
    const id = String(++this._requestId);
    const frame = { type, id, ...payload };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this._pending.delete(id);
        reject(new Error(`Request ${type} timed out`));
      }, timeoutMs);

      this._pending.set(id, { resolve, reject, timeout });
      this._ws.send(JSON.stringify(frame));
    });
  }

  /** Send a fire-and-forget frame. */
  send(type, payload = {}) {
    if (!this._connected) return;
    const id = String(++this._requestId);
    this._ws.send(JSON.stringify({ type, id, ...payload }));
  }

  // ── Internal ──

  _doConnect() {
    this._emit('status', 'connecting');

    try {
      this._ws = new WebSocket(this._url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      // Send connect frame with auth
      const connectFrame = {
        type: 'connect',
        role: 'operator',
        protocol_version: 1,
      };
      if (this._token) connectFrame.auth = this._token;
      this._ws.send(JSON.stringify(connectFrame));
    };

    this._ws.onmessage = (evt) => {
      let frame;
      try {
        frame = JSON.parse(evt.data);
      } catch {
        return;
      }
      this._handleFrame(frame);
    };

    this._ws.onclose = (evt) => {
      this._connected = false;
      this._emit('status', 'disconnected');
      // Reject all pending requests
      for (const [id, p] of this._pending) {
        clearTimeout(p.timeout);
        p.reject(new Error('Connection closed'));
      }
      this._pending.clear();
      if (!this._intentionalClose) this._scheduleReconnect();
    };

    this._ws.onerror = () => {
      // onclose will fire after this
    };
  }

  _handleFrame(frame) {
    const type = frame.type;

    // Handle hello-ok (connection established)
    if (type === 'hello-ok') {
      this._connected = true;
      this._reconnectDelay = 1000;
      this._methods = frame.methods || [];
      this._events = frame.events || [];
      this._emit('status', 'connected');
      this._emit('hello', frame);
      return;
    }

    // Handle errors
    if (type === 'error') {
      const id = frame.id;
      if (id && this._pending.has(id)) {
        const p = this._pending.get(id);
        clearTimeout(p.timeout);
        this._pending.delete(id);
        p.reject(new Error(frame.message || 'Unknown error'));
      }
      this._emit('error', frame);
      return;
    }

    // Handle correlated responses
    if (frame.id && this._pending.has(frame.id)) {
      const p = this._pending.get(frame.id);
      clearTimeout(p.timeout);
      this._pending.delete(frame.id);
      p.resolve(frame);
    }

    // Always emit as event for listeners
    this._emit(type, frame);
  }

  _emit(event, data) {
    const cbs = this._listeners.get(event);
    if (cbs) cbs.forEach(cb => {
      try { cb(data); } catch (e) { console.error('WS listener error:', e); }
    });
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    this._reconnectTimer = setTimeout(() => {
      this._doConnect();
    }, this._reconnectDelay);
    this._reconnectDelay = Math.min(this._reconnectDelay * 1.5, this._maxReconnectDelay);
  }
}

// Singleton
window.nanoWS = new NanobotWS();
