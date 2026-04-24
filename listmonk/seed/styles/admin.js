// Street Voices — Listmonk admin client-side customization
// Logo/icon base64 placeholders are substituted at seed time.
(function () {
  var LOGO_URI = '__LOGO_URI__';
  var COMPOSER_URL = '__COMPOSER_URL__';
  var YELLOW = '#ffd600';
  var BLACK = '#0a0a0a';

  function setFavicon() {
    document.querySelectorAll('link[rel*="icon"]').forEach(function (l) { l.parentNode.removeChild(l); });
    var link = document.createElement('link'); link.rel = 'icon'; link.type = 'image/png'; link.href = LOGO_URI;
    document.head.appendChild(link);
    var sc = document.createElement('link'); sc.rel = 'shortcut icon'; sc.type = 'image/png'; sc.href = LOGO_URI;
    document.head.appendChild(sc);
  }

  function brand() {
    var img = document.querySelector('.navbar-brand img') || document.querySelector('.navbar img');
    if (img && !img.dataset.svBranded) {
      img.dataset.svBranded = '1';
      img.src = LOGO_URI;
      img.removeAttribute('width'); img.removeAttribute('height');
      img.style.cssText = 'width:32px !important;height:32px !important;border-radius:0 !important;object-fit:contain !important;display:inline-block !important;vertical-align:middle !important;margin:0 !important;';
      var parent = img.parentNode;
      if (parent && !parent.querySelector('.sv-brand-text')) {
        var span = document.createElement('span');
        span.className = 'sv-brand-text';
        span.textContent = 'STREET VOICES';
        span.style.cssText = 'font-family:Rubik,Helvetica,Arial,sans-serif !important;font-size:16px !important;line-height:32px !important;font-weight:700 !important;letter-spacing:0.5px !important;text-transform:uppercase !important;color:#0a0a0a !important;display:inline-block !important;vertical-align:middle !important;margin-left:10px !important;';
        if (img.nextSibling) parent.insertBefore(span, img.nextSibling);
        else parent.appendChild(span);
      }
      var item = img.closest('a, .navbar-item') || parent;
      if (item) { item.style.display = 'flex'; item.style.alignItems = 'center'; item.style.gap = '0'; }
    }
    document.querySelectorAll('.navbar-brand img').forEach(function (i, idx) { if (idx > 0) i.style.display = 'none'; });
    document.querySelectorAll('.navbar-brand .navbar-item, .navbar-brand a').forEach(function (el) {
      Array.from(el.childNodes).forEach(function (n) {
        if (n.nodeType === 3 && /^\s*listmonk\s*$/i.test(n.textContent)) n.textContent = '';
      });
    });
    if (/listmonk/i.test(document.title) && !/Street Voices/i.test(document.title)) {
      document.title = document.title.replace(/listmonk/gi, 'Street Voices');
    }
  }

  function ensureAIFab() {
    var onCampaigns = /\/admin\/campaigns/i.test(location.hash + ' ' + location.pathname);
    var existing = document.getElementById('sv-ai-fab');
    if (!onCampaigns) { if (existing) existing.style.display = 'none'; return; }
    if (existing) { existing.style.display = 'flex'; return; }
    var fab = document.createElement('a');
    fab.id = 'sv-ai-fab';
    fab.href = COMPOSER_URL; fab.target = '_blank'; fab.rel = 'noopener';
    fab.innerHTML = '<span style="font-size:18px;line-height:1;">\u2728</span><span style="font-weight:700;font-size:13px;letter-spacing:0.4px;text-transform:uppercase;">AI Compose</span>';
    fab.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;align-items:center;gap:10px;padding:14px 20px;background:' + YELLOW + ';color:' + BLACK + ';border-radius:999px;text-decoration:none;font-family:Rubik,Helvetica,Arial,sans-serif;box-shadow:0 6px 20px rgba(255,214,0,0.45),0 2px 6px rgba(0,0,0,0.1);cursor:pointer;';
    document.body.appendChild(fab);
  }

  function isOnDashboard() {
    var p = location.pathname.replace(/\/$/, '');
    if (p !== '/admin' && !/\/admin\/dashboard$/.test(p)) return false;
    var h = location.hash;
    return h === '' || h === '#' || h === '#/' || h === '#/dashboard';
  }

  function fmt(n) { return Number(n || 0).toLocaleString('en-CA'); }
  function timeAgo(d) { if (!d) return '\u2014'; var s=(Date.now()-new Date(d).getTime())/1000; if (s<60) return 'just now'; if (s<3600) return Math.floor(s/60)+'m ago'; if (s<86400) return Math.floor(s/3600)+'h ago'; if (s<86400*7) return Math.floor(s/86400)+'d ago'; return new Date(d).toLocaleDateString('en-CA'); }

  async function fetchJSON(url) {
    var r = await fetch(url, { credentials: 'include' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return await r.json();
  }

  function cardShell(title, contentHtml) {
    return '<div style="background:#fff;border:1px solid #eeeae0;border-radius:16px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.04);font-family:Rubik,sans-serif;min-width:0;height:100%;display:flex;flex-direction:column;">'
      + '<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-shrink:0;">'
      + '<span style="display:inline-block;width:6px;height:18px;background:' + YELLOW + ';border-radius:3px;"></span>'
      + '<h3 style="margin:0;font-size:13px;font-weight:700;color:#777;text-transform:uppercase;letter-spacing:0.7px;">' + title + '</h3>'
      + '</div><div style="flex:1;min-height:0;">' + contentHtml + '</div></div>';
  }

  async function overviewContent() {
    try {
      var data = await fetchJSON('/api/dashboard/counts');
      var d = data.data || {}; var L = d.lists || {}; var S = d.subscribers || {}; var C = d.campaigns || {}; var M = d.messages || 0;
      var statuses = C.by_status || {};
      var statusLine = Object.keys(statuses).map(function (k) { return fmt(statuses[k]) + ' ' + k; }).join(' \u00b7 ') || '\u2014';
      var metric = function (icon, value, label, sub, noBorder) {
        return '<div style="padding:12px 0;' + (noBorder ? '' : 'border-bottom:1px solid #eeeae0;') + '">'
          + '<div style="display:flex;align-items:center;gap:14px;">'
          + '<div style="width:36px;height:36px;background:#fffbe6;border:1px solid ' + YELLOW + ';border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">' + icon + '</div>'
          + '<div style="flex:1;min-width:0;">'
          + '<div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;">'
          + '<div style="font-size:24px;font-weight:800;color:#0a0a0a;letter-spacing:-0.5px;line-height:1;">' + value + '</div>'
          + (sub ? '<div style="font-size:10px;color:#999;text-align:right;font-weight:500;">' + sub + '</div>' : '')
          + '</div>'
          + '<div style="font-size:10px;color:#777;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;margin-top:4px;">' + label + '</div>'
          + '</div></div></div>';
      };
      return metric('\ud83d\udccb', fmt(L.total), 'Lists', fmt(L.public || 0) + ' public \u00b7 ' + fmt(L.private || 0) + ' private')
        + metric('\ud83d\udc65', fmt(S.total), 'Subscribers', fmt(S.blocklisted || 0) + ' blocklisted \u00b7 ' + fmt(S.orphans || 0) + ' orphans')
        + metric('\ud83d\ude80', fmt(C.total), 'Campaigns', statusLine)
        + metric('\ud83d\udce7', fmt(M), 'Messages sent', '', true);
    } catch (e) { return '<div style="color:#b22020;font-size:12px;">Could not load: ' + e.message + '</div>'; }
  }

  function quickActionsContent() {
    var btn = function (icon, label, href, primary, external) {
      var bg = primary ? YELLOW : '#fff'; var color = primary ? BLACK : '#2d2d2d'; var border = primary ? YELLOW : '#eeeae0';
      return '<a href="' + href + '" target="' + (external ? '_blank' : '_self') + '" style="display:flex;align-items:center;gap:12px;padding:13px 16px;background:' + bg + ';color:' + color + ';border:1px solid ' + border + ';border-radius:12px;text-decoration:none;font-weight:600;font-size:14px;"><span style="font-size:18px;width:22px;text-align:center;">' + icon + '</span><span>' + label + '</span></a>';
    };
    return '<div style="display:flex;flex-direction:column;gap:10px;">'
      + btn('\u2728', 'Open AI Composer', COMPOSER_URL, true, true)
      + btn('+', 'New campaign', '/admin/campaigns/new', false, false)
      + btn('\ud83d\udc64', 'Add subscriber', '/admin/subscribers/new', false, false)
      + btn('\ud83d\udccb', 'Manage lists', '/admin/lists', false, false)
      + '</div>';
  }

  async function lastWeekContent() {
    try {
      var data = await fetchJSON('/api/campaigns?per_page=100');
      var camps = (data && data.data && data.data.results) || [];
      var seven = Date.now() - 7 * 24 * 3600 * 1000;
      var recent = camps.filter(function (c) { var d = c.started_at || c.updated_at; return d && new Date(d).getTime() > seven; });
      var t = recent.reduce(function (a, c) { a.sent += c.sent || 0; a.views += c.views || 0; a.clicks += c.clicks || 0; return a; }, { sent: 0, views: 0, clicks: 0 });
      var openRate = t.sent ? (100 * t.views / t.sent).toFixed(1) + '%' : '\u2014';
      var clickRate = t.sent ? (100 * t.clicks / t.sent).toFixed(1) + '%' : '\u2014';
      var stat = function (label, value, sub) {
        return '<div style="text-align:center;padding:14px 8px;"><div style="font-size:30px;font-weight:800;color:#0a0a0a;letter-spacing:-0.5px;line-height:1.1;">' + value + '</div><div style="font-size:10px;color:#777;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;margin-top:4px;">' + label + '</div>' + (sub ? '<div style="font-size:12px;color:#aaa;margin-top:3px;font-weight:500;">' + sub + '</div>' : '') + '</div>';
      };
      return '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;">' + stat('Sent', fmt(t.sent)) + stat('Opens', fmt(t.views), openRate) + stat('Clicks', fmt(t.clicks), clickRate) + '</div><div style="margin-top:16px;padding-top:14px;border-top:1px solid #eeeae0;text-align:center;font-size:11px;color:#aaa;">across ' + recent.length + ' campaign' + (recent.length === 1 ? '' : 's') + ' in the last 7 days</div>';
    } catch (e) { return '<div style="color:#b22020;font-size:12px;">Could not load: ' + e.message + '</div>'; }
  }

  var SKELETON_SVG = '<svg viewBox="0 0 400 180" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:180px;"><defs><linearGradient id="svChartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#ffd600" stop-opacity="0.35"/><stop offset="100%" stop-color="#ffd600" stop-opacity="0"/></linearGradient></defs><g stroke="#eeeae0" stroke-width="1"><line x1="0" y1="30" x2="400" y2="30" stroke-dasharray="3 4"/><line x1="0" y1="75" x2="400" y2="75" stroke-dasharray="3 4"/><line x1="0" y1="120" x2="400" y2="120" stroke-dasharray="3 4"/><line x1="0" y1="165" x2="400" y2="165"/></g><path d="M0,140 C60,120 90,90 140,85 S210,110 260,80 S340,50 400,60 L400,180 L0,180 Z" fill="url(#svChartFill)"/><path d="M0,140 C60,120 90,90 140,85 S210,110 260,80 S340,50 400,60" fill="none" stroke="#ffd600" stroke-width="2" opacity="0.55"/><g fill="#ffd600" opacity="0.7"><circle cx="0" cy="140" r="3"/><circle cx="70" cy="108" r="3"/><circle cx="140" cy="85" r="3"/><circle cx="210" cy="100" r="3"/><circle cx="260" cy="80" r="3"/><circle cx="330" cy="62" r="3"/><circle cx="400" cy="60" r="3"/></g></svg>';

  var CAMPAIGN_VIEWS_EMPTY = '<div class="sv-chart-slot" style="display:flex;flex-direction:column;height:100%;min-height:240px;"><div style="flex:1;position:relative;margin-top:-4px;">' + SKELETON_SVG + '</div><div style="text-align:center;padding:14px 8px 4px;border-top:1px dashed #eeeae0;margin-top:6px;"><div style="font-size:13px;font-weight:700;color:#0a0a0a;margin-bottom:4px;">No campaign views yet</div><div style="font-size:11px;color:#999;line-height:1.5;">Send your first campaign to see opens over time here.</div></div></div>';

  function tryRelocateChart() {
    var target = document.getElementById('sv-cell-campaigns');
    if (!target) return;
    var slot = target.querySelector('.sv-chart-slot');
    if (!slot) return;
    if (slot.querySelector('canvas, svg.chartjs-render-monitor, .b-tabs')) return;
    var candidates = document.querySelectorAll('h4, h3, .title, .subtitle, header');
    var chartHeader = null;
    for (var i = 0; i < candidates.length; i++) {
      var txt = (candidates[i].textContent || '').trim().toLowerCase();
      if (txt === 'campaign views') { chartHeader = candidates[i]; break; }
    }
    if (!chartHeader) return;
    var container = chartHeader;
    while (container && container.parentNode) {
      if (container.querySelector && container.querySelector('canvas, svg')) break;
      container = container.parentNode;
    }
    if (!container || container === document.body || container.id === 'sv-dashboard') return;
    slot.appendChild(container);
  }

  async function ensureWidgets() {
    if (!isOnDashboard()) {
      document.body.classList.remove('sv-on-dashboard');
      var ex = document.getElementById('sv-dashboard');
      if (ex) ex.remove();
      return;
    }
    document.body.classList.add('sv-on-dashboard');
    if (document.getElementById('sv-dashboard')) { tryRelocateChart(); return; }
    var host = (document.querySelector('.dashboard') || {}).parentNode
      || document.querySelector('main.column')
      || document.querySelector('section.column:not(.is-2):not(.is-3):not(.is-4):not(.is-narrow)')
      || document.querySelector('.column:not(.is-2):not(.is-3):not(.is-4):not(.is-narrow)')
      || document.querySelector('section.section > .container, section.section > div')
      || document.querySelector('section.section')
      || document.querySelector('main');
    if (!host) return;

    var wrap = document.createElement('div');
    wrap.id = 'sv-dashboard';
    wrap.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:20px;width:100%;padding:8px 0;';
    var today = new Date().toLocaleDateString('en-CA', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
    var heading = document.createElement('div');
    heading.style.cssText = 'grid-column:1/-1;margin:0 0 4px 0;font-family:Rubik,sans-serif;';
    heading.innerHTML = '<div style="font-size:12px;color:#777;text-transform:uppercase;letter-spacing:0.7px;font-weight:600;">' + today + '</div><h1 style="margin:4px 0 0 0;font-size:28px;font-weight:800;color:#0a0a0a;letter-spacing:-0.5px;">Dashboard</h1>';
    wrap.appendChild(heading);

    var cells = [
      { id: 'sv-cell-overview', title: 'Overview', loader: overviewContent },
      { id: 'sv-cell-actions', title: 'Quick actions', html: quickActionsContent() },
      { id: 'sv-cell-campaigns', title: 'Campaign views', html: CAMPAIGN_VIEWS_EMPTY },
      { id: 'sv-cell-week', title: 'Last 7 days', loader: lastWeekContent },
    ];
    cells.forEach(function (cell) {
      var d = document.createElement('div');
      d.id = cell.id; d.style.cssText = 'min-width:0;';
      d.innerHTML = cardShell(cell.title, cell.html || '<div style="color:#777;font-size:12px;">Loading\u2026</div>');
      wrap.appendChild(d);
    });
    host.insertBefore(wrap, host.firstChild);
    cells.forEach(function (cell) {
      if (!cell.loader) return;
      cell.loader().then(function (html) {
        var s = document.getElementById(cell.id);
        if (s) s.innerHTML = cardShell(cell.title, html);
      });
    });
    if (window.innerWidth < 900) wrap.style.gridTemplateColumns = '1fr';
    tryRelocateChart();
  }

  // Kill residual blue accents
  function isBlueish(rgb) {
    if (!rgb) return false;
    var m = rgb.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (!m) return false;
    var r = +m[1], g = +m[2], b = +m[3];
    return b > 130 && b > r + 30 && b > g + 30 && r < 120;
  }
  function killBlues() {
    document.querySelectorAll('*').forEach(function (el) {
      if (el.dataset.svBlueKilled || el.id === 'sv-ai-fab' || el.closest('#sv-dashboard')) return;
      var s = window.getComputedStyle(el); var changed = false;
      if (isBlueish(s.backgroundColor)) { el.style.setProperty('background-color', YELLOW, 'important'); el.style.setProperty('color', BLACK, 'important'); changed = true; }
      ['Top','Right','Bottom','Left'].forEach(function (side) {
        if (isBlueish(s['border' + side + 'Color'])) { el.style.setProperty('border-' + side.toLowerCase() + '-color', YELLOW, 'important'); changed = true; }
      });
      if (changed) el.dataset.svBlueKilled = '1';
    });
  }

  function run() { setFavicon(); brand(); killBlues(); ensureAIFab(); ensureWidgets(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
  else run();
  setInterval(run, 1200);
  window.addEventListener('hashchange', run);
  window.addEventListener('popstate', run);
  window.addEventListener('resize', function () {
    var wrap = document.getElementById('sv-dashboard');
    if (wrap) wrap.style.gridTemplateColumns = window.innerWidth < 900 ? '1fr' : '1fr 1fr';
  });
})();
