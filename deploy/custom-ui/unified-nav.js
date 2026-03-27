/**
 * Street Voices — Unified Navigation Bar
 * Works on BOTH LibreChat and Paperclip pages (served from the same origin via nginx).
 * Detects current context from URL path and highlights the right tab.
 */
(function () {
  'use strict';

  var NAV_ID = 'sv-unified-nav';
  var APPS = [
    { key: 'chat',       label: 'Chat',            url: '/',               icon: 'chat' },
    { key: 'marketplace',label: 'Marketplace',      url: '/agents', icon: 'store' },
  ];

  var ICONS = {
    chat: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    grid: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    store: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    users: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
  };

  function injectStyles() {
    if (document.getElementById('sv-nav-styles')) return;
    var style = document.createElement('style');
    style.id = 'sv-nav-styles';
    style.textContent =
      '#' + NAV_ID + ' {' +
        'position:fixed;top:0;left:0;right:0;z-index:9999;' +
        'height:40px;background:transparent;border-bottom:none;' +
        'display:flex;align-items:center;padding:0 16px;gap:2px;' +
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;' +
        'font-size:13px;user-select:none;-webkit-user-select:none;' +
      '}' +
      '#' + NAV_ID + ' .sv-brand{font-weight:700;font-size:13px;letter-spacing:-0.3px;color:#ff6b35;margin-right:20px;white-space:nowrap;display:flex;align-items:center;gap:6px;}' +
      '#' + NAV_ID + ' .sv-brand .dot{width:6px;height:6px;border-radius:50%;background:#22c55e;box-shadow:0 0 4px #22c55e;animation:svPulse 2s infinite;}' +
      '@keyframes svPulse{0%,100%{opacity:1}50%{opacity:.4}}' +
      '#' + NAV_ID + ' .sv-tab{display:flex;align-items:center;gap:6px;padding:6px 12px;border-radius:6px;color:#8888a0;cursor:pointer;transition:all .15s;text-decoration:none;font-size:12px;font-weight:500;white-space:nowrap;}' +
      '#' + NAV_ID + ' .sv-tab:hover{background:rgba(255,255,255,.06);color:#e8e8ed;}' +
      '#' + NAV_ID + ' .sv-tab.active{background:rgba(255,107,53,.1);color:#ff6b35;}' +
      '#' + NAV_ID + ' .sv-tab svg{opacity:.7;flex-shrink:0;}' +
      '#' + NAV_ID + ' .sv-tab.active svg{opacity:1;}' +
      '#' + NAV_ID + ' .sv-spacer{flex:1;}' +
      '#' + NAV_ID + ' .sv-status{font-size:11px;color:#555;font-family:"SF Mono","Menlo",monospace;}' +
      'body.sv-sb-page [aria-label="Toggle theme"]{display:none !important;}' +
      'body.sv-sb-page a[href="/settings"]{display:none !important;}' +
      'body.sv-nav-active{padding-top:0 !important;}' +
      'body.sv-nav-active #root{height:100vh !important;margin-top:0 !important;}' +
      'body.sv-nav-active nav#chat-history-nav{top:0 !important;height:100vh !important;}' +
      '@media (max-width:640px){' +
        '#' + NAV_ID + ' .sv-brand span{display:none;}' +
        '#' + NAV_ID + ' .sv-tab span{display:none;}' +
        '#' + NAV_ID + ' .sv-tab{padding:6px 8px;}' +
        '#' + NAV_ID + ' .sv-status{display:none;}' +
      '}';
    document.head.appendChild(style);
  }

  function getActiveKey() {
    var path = window.location.pathname;
    if (/^\/[A-Z]{2,6}\//.test(path)) return 'mc';
    if (path.startsWith('/marketplace')) return 'marketplace';
    if (path.startsWith('/social')) return 'social';
    return 'chat';
  }

  function createNav() {
    if (document.getElementById(NAV_ID)) return;
    injectStyles();
    var nav = document.createElement('div');
    nav.id = NAV_ID;
    nav.style.display = 'none';
    document.body.prepend(nav);
    document.body.classList.add('sv-nav-active');
  }

  function init() { createNav(); updateSbPageClass(); }

  // Detect streetbot (non-home) pages and toggle body class to hide theme/profile controls
  var SB_PATHS = ['/jobs','/agents','/groups','/news','/gallery','/directory','/social',
    '/how-it-works','/about','/terms','/privacy','/manage','/notifications','/academy','/learning','/profile','/street-profile','/messages','/accounting'];
  function isStreetbotPage() {
    var p = window.location.pathname;
    for (var i = 0; i < SB_PATHS.length; i++) {
      if (p === SB_PATHS[i] || p.startsWith(SB_PATHS[i] + '/')) return true;
    }
    // Also match /STR/ (Paperclip) pages
    if (/^\/[A-Z]{2,6}\//.test(p)) return true;
    return false;
  }
  function updateSbPageClass() {
    if (isStreetbotPage()) {
      document.body.classList.add('sv-sb-page');
    } else {
      document.body.classList.remove('sv-sb-page');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { setTimeout(init, 300); });
  } else { setTimeout(init, 300); }
  setInterval(function () { if (!document.getElementById(NAV_ID)) createNav(); updateSbPageClass(); }, 2000);

  // ── If inside iframe (embed mode), hide the entire sidebar ──
  if (window !== window.top || window.location.search.indexOf('embed=true') !== -1) {
    var embedStyle = document.createElement('style');
    embedStyle.id = 'sv-embed-hide';
    embedStyle.textContent = [
      'nav#chat-history-nav { display: none !important; }',
      '.nav { display: none !important; }',
      'div.flex-shrink-0:has(> .nav) { display: none !important; width: 0 !important; }',
      '#sv-standalone-sidebar { display: none !important; }',
      '#root > div.flex { margin-left: 0 !important; }',
    ].join('\n');
    document.head.appendChild(embedStyle);
  }

  // ── Immediate CSS hiding (runs before sidebar detection) ──
  (function injectEarlyHideCSS() {
    if (document.getElementById('sv-early-hide')) return;
    var s = document.createElement('style');
    s.id = 'sv-early-hide';
    s.textContent = [
      '[role="contentinfo"] { display: none !important; }',
      'a[href="https://librechat.ai"] { display: none !important; }',
      'nav[aria-label="Controls"] { display: none !important; }',
      'button[aria-label="Open Control Panel"] { display: none !important; }',
      'button[aria-label="Close Control Panel"] { display: none !important; }',
      'button[aria-label="Close right side panel"] { display: none !important; }',
      'button[aria-label="Add multi-conversation"] { display: none !important; }',
      '[data-testid="add-multi-convo-button"] { display: none !important; }',
      'button[aria-label="Tools Options"] { display: none !important; }',
      'a[href="/agents"] { display: none !important; }',
      '[data-testid="nav-agents-marketplace-button"] { display: none !important; }',
      'button[aria-label="Bookmarks"] { display: none !important; }',
      'button[aria-label="New chat"] { display: none !important; }',
      '#sv-new-chat-btn { display: flex !important; }',
      '[data-panel-id="controls-nav"] { display: none !important; }',
      '.sidenav-mask { display: none !important; }',
      '#toggle-right-nav { display: none !important; }',
      '#toggle-right-nav + * { display: none !important; }',
      'div[role="separator"].group { display: none !important; }',
      'html:not(.dark) .sv-logo-img { filter: brightness(0) !important; }',
      '.dark .sv-logo-img { filter: none !important; }',
      '.dark .sv-sidebar-icon { filter: brightness(0) invert(1) !important; opacity: 1 !important; }',
      'html:not(.dark) .sv-sidebar-icon { filter: brightness(0) !important; }',
      '.dark #sv-sidebar-static svg { color: white !important; }',
      '.dark nav#chat-history-nav svg { color: white !important; }',
      '.dark img[src*="mdi/svg"] { filter: brightness(0) invert(1) !important; }',
      '.dark #sv-sidebar-nav-wrap img { filter: brightness(0) invert(1) !important; opacity: 1 !important; }',
    ].join(' ');
    document.head.appendChild(s);
  })();

  // ── Sidebar buttons ──
  var MC_ID = 'sv-sidebar-mc';
  var SOCIAL_ID = 'sv-sidebar-social';
  var MC_SVG = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>';
  var SOCIAL_SVG = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>';

  function createSidebarBtn(id, label, icon, url) {
    var btn = document.createElement('div');
    btn.id = id;
    btn.setAttribute('role', 'button');
    btn.setAttribute('tabindex', '0');
    btn.setAttribute('aria-label', label);
    btn.style.cssText = 'position:relative;display:flex;width:100%;cursor:pointer;align-items:center;justify-content:space-between;border-radius:0.5rem;padding:0.5rem 0.75rem;font-size:0.875rem;line-height:1.25rem;outline:none;color:var(--text-primary);';
    btn.innerHTML =
      '<div style="display:flex;flex:1;align-items:center;overflow:hidden;padding-right:1.5rem">' +
        '<div style="margin-right:0.5rem;width:1.25rem;height:1.25rem;flex-shrink:0">' + icon + '</div>' +
        '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + label + '</span>' +
      '</div>';
    btn.addEventListener('mouseenter', function () { btn.style.background = 'var(--surface-active-alt, rgba(255,255,255,0.06))'; });
    btn.addEventListener('mouseleave', function () { btn.style.background = ''; });
    btn.addEventListener('click', function (e) { e.stopPropagation(); window.location.href = url; }, true);
    return btn;
  }

  function injectSidebarButtons() {
    try { _injectSidebarButtonsInner(); } catch (e) {
      // Swallow DOM errors — React may be mid-reconciliation
    }
  }

  function _injectSidebarButtonsInner() {
    // All traffic goes through :3180 nginx proxy now

    // Deduplicate
    [MC_ID, SOCIAL_ID].forEach(function (id) {
      var all = document.querySelectorAll('#' + id);
      for (var k = 1; k < all.length; k++) all[k].remove();
    });

    var nav = document.getElementById('chat-history-nav');
    if (!nav) return;
    var outerWrapper = nav.firstElementChild;
    if (!outerWrapper || !outerWrapper.classList.contains('flex-1')) return;

    var scrollContainer = null;
    for (var i = 0; i < outerWrapper.children.length; i++) {
      var child = outerWrapper.children[i];
      if (child.classList.contains('flex-grow') && child.classList.contains('min-h-0')) {
        scrollContainer = child; break;
      }
    }
    if (!scrollContainer) return;

    var MKT_ID = 'sv-sidebar-mkt';

    // Hide original marketplace button
    if (!document.getElementById('sv-hide-orig-mkt')) {
      var hideCSS = document.createElement('style');
      hideCSS.id = 'sv-hide-orig-mkt';
      hideCSS.textContent = 'a[href="/agents"], [data-testid="nav-agents-marketplace-button"] { display: none !important; } [data-panel-id="controls-nav"] { display: none !important; } .sidenav-mask { display: none !important; } button[aria-label="Bookmarks"] { display: none !important; } button[aria-label="New chat"] { display: none !important; } #sv-new-chat-btn { display: flex !important; } button[aria-label="Open Control Panel"] { display: none !important; } button[aria-label="Close Control Panel"] { display: none !important; } button[aria-label="Close right side panel"] { display: none !important; } [data-testid="mobile-header-new-chat-button"] { display: none !important; } button[aria-label="Tools Options"] { display: none !important; } button[aria-label="Add multi-conversation"] { display: none !important; } button[title="Add multi-conversation"] { display: none !important; } [data-testid="multi-convo-button"] { display: none !important; } [data-testid="add-multi-convo-button"] { display: none !important; } div[role="contentinfo"] { display: none !important; } div[role="contentinfo"] + * { display: none !important; } a[href="https://librechat.ai"] { display: none !important; } nav[aria-label="Controls"] { display: none !important; } button[aria-label="Open Control Panel"] + hr { display: none !important; } [role="contentinfo"] { display: none !important; }';
      document.head.appendChild(hideCSS);
    }

    // ── Single-scroll sidebar: icons scroll away, then chats scroll ──
    if (!document.getElementById('sv-single-scroll')) {
      var ssCSS = document.createElement('style');
      ssCSS.id = 'sv-single-scroll';
      ssCSS.textContent = [
        // Outer wrapper: ONE scrollbar for the entire sidebar, visible on hover
        'nav#chat-history-nav > .flex-1.flex-col { overflow-y: auto !important; overflow-x: hidden !important; display: flex !important; flex-direction: column !important; scrollbar-width: thin !important; scrollbar-color: transparent transparent !important; }',
        'nav#chat-history-nav:hover > .flex-1.flex-col { scrollbar-color: rgba(128,128,128,0.3) transparent !important; }',
        'nav#chat-history-nav > .flex-1.flex-col::-webkit-scrollbar { width: 4px !important; background: transparent !important; }',
        'nav#chat-history-nav > .flex-1.flex-col::-webkit-scrollbar-track { background: transparent !important; }',
        'nav#chat-history-nav > .flex-1.flex-col::-webkit-scrollbar-thumb { background: transparent !important; border-radius: 4px !important; }',
        '.dark nav#chat-history-nav:hover > .flex-1.flex-col::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15) !important; }',
        'html:not(.dark) nav#chat-history-nav:hover > .flex-1.flex-col::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15) !important; }',
        // Default order for unmanaged children
        'nav#chat-history-nav > .flex-1.flex-col > * { order: 3; }',
        '#sv-sidebar-header { order: 1 !important; }',
        '#sv-sidebar-static { order: 2 !important; }',
        '#sv-sidebar-nav-wrap { order: 4 !important; flex-shrink: 0 !important; max-height: 240px !important; overflow-y: auto !important; overflow-x: hidden !important; scrollbar-width: none !important; mask-image: linear-gradient(to bottom, black 80%, transparent 100%) !important; -webkit-mask-image: linear-gradient(to bottom, black 80%, transparent 100%) !important; }',
        '#sv-sidebar-nav-wrap::-webkit-scrollbar { width: 0 !important; display: none !important; }',
        // Pin the SV header at top while everything else scrolls
        '#sv-sidebar-header { position: sticky !important; top: 0 !important; z-index: 10 !important; background: var(--surface-primary-alt, #fff) !important; }',
        // Chat section: fill remaining space, hide ITS scrollbar (outer handles it)
        '.flex.min-h-0.flex-grow.flex-col.overflow-hidden { flex: 1 1 auto !important; min-height: 200px !important; overflow: hidden !important; }',
        // Hide ALL internal scrollbars — only the outer wrapper scrollbar shows
        '.flex.min-h-0.flex-grow.flex-col.overflow-hidden * { scrollbar-width: none !important; }',
        '.flex.min-h-0.flex-grow.flex-col.overflow-hidden *::-webkit-scrollbar { width: 0 !important; display: none !important; }',
      ].join('\n');
      document.head.appendChild(ssCSS);
    }

    // Forward scroll events: when outer sidebar finishes scrolling icons,
    // redirect wheel events into the chat section's virtual list
    if (!window._svScrollForward) {
      window._svScrollForward = true;

      // --- Custom unified scrollbar ---
      function initCustomScrollbar() {
        var nav = document.querySelector('nav#chat-history-nav');
        if (!nav || document.getElementById('sv-custom-scrollbar')) return;
        var track = document.createElement('div');
        track.id = 'sv-custom-scrollbar';
        track.innerHTML = '<div id="sv-custom-thumb"></div>';
        nav.style.position = 'relative';
        nav.appendChild(track);

        var trackStyle = track.style;
        trackStyle.cssText = 'position:absolute;top:0;right:0;width:6px;height:100%;z-index:100;opacity:0;transition:opacity 0.2s;pointer-events:none;';
        var thumb = document.getElementById('sv-custom-thumb');
        thumb.style.cssText = 'position:absolute;top:0;right:0;width:4px;margin:0 1px;border-radius:4px;background:rgba(128,128,128,0.4);transition:top 0.05s,height 0.05s;pointer-events:auto;cursor:pointer;';

        // Show on hover
        nav.addEventListener('mouseenter', function() { track.style.opacity = '1'; });
        nav.addEventListener('mouseleave', function() { track.style.opacity = '0'; });

        function getScrollers() {
          var navWrap = document.getElementById('sv-sidebar-nav-wrap');
          var chatScroller = document.querySelector('nav#chat-history-nav .ReactVirtualized__List');
          return { navWrap: navWrap, chat: chatScroller };
        }

        function updateThumb() {
          var s = getScrollers();
          if (!s.navWrap) return;
          var navTotal = s.navWrap.scrollHeight;
          var navVisible = s.navWrap.clientHeight;
          var navScroll = s.navWrap.scrollTop;
          var chatTotal = s.chat ? s.chat.scrollHeight : 0;
          var chatVisible = s.chat ? s.chat.clientHeight : 0;
          var chatScroll = s.chat ? s.chat.scrollTop : 0;
          var totalContent = navTotal + chatTotal;
          var totalVisible = navVisible + chatVisible;
          var totalScroll = navScroll + chatScroll;
          if (totalContent <= totalVisible) { thumb.style.height = '0'; return; }
          var trackH = nav.clientHeight;
          var thumbH = Math.max(30, (totalVisible / totalContent) * trackH);
          var thumbTop = (totalScroll / (totalContent - totalVisible)) * (trackH - thumbH);
          thumb.style.height = thumbH + 'px';
          thumb.style.top = thumbTop + 'px';
        }

        // Update on scroll
        var navWrap = document.getElementById('sv-sidebar-nav-wrap');
        if (navWrap) navWrap.addEventListener('scroll', updateThumb);
        var chatCheck = setInterval(function() {
          var c = document.querySelector('nav#chat-history-nav .ReactVirtualized__List');
          if (c && !c._svScrollBound) {
            c._svScrollBound = true;
            c.addEventListener('scroll', updateThumb);
            clearInterval(chatCheck);
            updateThumb();
          }
        }, 500);
        setInterval(updateThumb, 1000); // periodic refresh

        // Drag support
        var dragging = false, dragStartY = 0, dragStartScroll = 0;
        thumb.addEventListener('mousedown', function(e) {
          dragging = true; dragStartY = e.clientY;
          var s = getScrollers();
          dragStartScroll = (s.navWrap ? s.navWrap.scrollTop : 0) + (s.chat ? s.chat.scrollTop : 0);
          e.preventDefault();
        });
        document.addEventListener('mousemove', function(e) {
          if (!dragging) return;
          var s = getScrollers();
          var navTotal = s.navWrap ? s.navWrap.scrollHeight : 0;
          var navVisible = s.navWrap ? s.navWrap.clientHeight : 0;
          var chatTotal = s.chat ? s.chat.scrollHeight : 0;
          var chatVisible = s.chat ? s.chat.clientHeight : 0;
          var totalContent = navTotal + chatTotal;
          var totalVisible = navVisible + chatVisible;
          var trackH = nav.clientHeight;
          var thumbH = Math.max(30, (totalVisible / totalContent) * trackH);
          var dy = e.clientY - dragStartY;
          var scrollRatio = dy / (trackH - thumbH);
          var targetScroll = dragStartScroll + scrollRatio * (totalContent - totalVisible);
          // Distribute scroll between navWrap and chat
          if (s.navWrap) {
            var navMax = navTotal - navVisible;
            if (targetScroll <= navMax) {
              s.navWrap.scrollTop = targetScroll;
              if (s.chat) s.chat.scrollTop = 0;
            } else {
              s.navWrap.scrollTop = navMax;
              if (s.chat) s.chat.scrollTop = targetScroll - navMax;
            }
          }
          e.preventDefault();
        });
        document.addEventListener('mouseup', function() { dragging = false; });
      }
      setTimeout(initCustomScrollbar, 1000);

      document.addEventListener('wheel', function(e) {
        var outer = document.querySelector('nav#chat-history-nav > .flex-1.flex-col');
        if (!outer) return;
        // Only intercept scrolls within the sidebar
        if (!outer.contains(e.target) && e.target !== outer) return;
        var chatScroller = document.querySelector('nav#chat-history-nav .ReactVirtualized__List');
        if (!chatScroller) return;
        var navWrap = document.getElementById('sv-sidebar-nav-wrap');
        var navAtBottom = navWrap && (navWrap.scrollTop + navWrap.clientHeight >= navWrap.scrollHeight - 2);
        var navAtTop = navWrap && navWrap.scrollTop <= 0;
        if (navAtBottom && e.deltaY > 0) {
          // Nav-wrap can't scroll down more — forward to chat list
          chatScroller.scrollTop += e.deltaY;
          e.preventDefault();
        } else if (chatScroller.scrollTop > 0 && e.deltaY < 0) {
          // Chat is scrolled — scroll it back up first before scrolling outer
          chatScroller.scrollTop += e.deltaY;
          if (chatScroller.scrollTop < 0) chatScroller.scrollTop = 0;
          e.preventDefault();
        } else if (navWrap && e.deltaY > 0) {
          navWrap.scrollTop += e.deltaY;
          e.preventDefault();
        } else if (navWrap && e.deltaY < 0 && !navAtTop) {
          navWrap.scrollTop += e.deltaY;
          e.preventDefault();
        }
      }, { passive: false });
    }

    // Hide MCP Servers, Add multi-conversation, Open Control Panel, footer
    document.querySelectorAll('button').forEach(function(b) {
      var txt = b.textContent.trim();
      if (txt === 'MCP Servers') b.style.display = 'none';
      if (txt === 'Add multi-conversation' || b.getAttribute('title') === 'Add multi-conversation') b.style.display = 'none';
      if (txt === 'Open Control Panel' || txt === 'Close Control Panel') b.style.display = 'none';
    });
    // Also hide by tooltip — the + circle button
    document.querySelectorAll('button[class*="multi"], button[data-testid*="multi"]').forEach(function(b) {
      b.style.display = 'none';
    });
    // Hide LibreChat footer
    document.querySelectorAll('[role="contentinfo"]').forEach(function(el) {
      el.style.display = 'none';
      if (el.parentElement) el.parentElement.style.display = 'none';
    });
    document.querySelectorAll('a[href="https://librechat.ai"]').forEach(function(a) {
      var p = a.closest('[role="contentinfo"]') || a.parentElement;
      if (p) { p.style.display = 'none'; if (p.parentElement) p.parentElement.style.display = 'none'; }
    });
    // Hide right-side Controls panel + Open Control Panel button
    var controlsNav = document.querySelector('nav[aria-label="Controls"]');
    if (controlsNav) controlsNav.style.display = 'none';
    document.querySelectorAll('button[aria-label="Open Control Panel"], button[aria-label="Close right side panel"]').forEach(function(b) {
      b.style.display = 'none';
    });

    // Clean up old scroll fix styles
    ['sv-scroll-fix-v2', 'sv-scroll-fix', 'sv-unified-scroll-fix', 'sv-hide-wrap-scrollbar'].forEach(function(id) {
      var el = document.getElementById(id); if (el) el.remove();
    });

    // ── Replace sidebar header with SV logo + collapse icon ──
    var svHeader = document.getElementById('sv-sidebar-header');
    if (!svHeader) {
      var origHeader = outerWrapper.children[0];
      if (origHeader) {
        // Grab the original close sidebar button's click handler
        var closeSidebarBtn = origHeader.querySelector('button[aria-label="Close sidebar"]');
        svHeader = document.createElement('div');
        svHeader.id = 'sv-sidebar-header';
        svHeader.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:0.5rem 1.25rem 0.5rem 0.75rem;width:100%;';
        // Logo + text
        var logoWrap = document.createElement('div');
        logoWrap.style.cssText = 'display:flex;align-items:center;gap:10px;cursor:pointer;margin-top:4px;';
        logoWrap.innerHTML = '<img class="sv-logo-img" src="/images/streetbot/megaphone-icon.svg" alt="Street Voices" width="32" height="32" style="width:32px;height:32px;min-width:28px;flex-shrink:0;" /><img class="sv-logo-img" src="/images/streetbot/streetvoices-text.svg" alt="Street Voices" width="130" height="19" style="max-width:100%;height:auto;" />';
        logoWrap.addEventListener('click', function() { window.location.href = '/'; });
        svHeader.appendChild(logoWrap);
        // Custom collapse button (React keeps recreating the native one, so we make our own)
        var collapseBtn = document.createElement('button');
        collapseBtn.id = 'sv-header-collapse';
        collapseBtn.title = 'Close sidebar';
        collapseBtn.style.cssText = 'cursor:pointer;display:flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:0.5rem;border:none;background:transparent;color:var(--text-primary);margin-left:auto;flex-shrink:0;';
        collapseBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>';
        collapseBtn.addEventListener('click', function(e) {
          e.stopPropagation();
          e.preventDefault();
          // Trigger collapse via the setupCollapse mechanism
          var evt = new CustomEvent('sv-collapse-sidebar');
          document.dispatchEvent(evt);
        });
        svHeader.appendChild(collapseBtn);
        origHeader.style.display = 'none';
        outerWrapper.insertBefore(svHeader, origHeader);
      }
    }

    var WRAP_ID = 'sv-sidebar-nav-wrap';
    var MKT_SVG = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>';

    var SB_BASE = '';
    var STATIC_ID = 'sv-sidebar-static';
    var homeIcon = '<img src="/images/sidebar-icons/home.svg" alt="" width="20" height="20" class="sv-sidebar-icon" style="opacity:0.7;flex-shrink:0;" />';
    var notifIcon = '<img src="/images/sidebar-icons/notifications.svg" alt="" width="20" height="20" class="sv-sidebar-icon" style="opacity:0.7;flex-shrink:0;" />';

    // ── STATIC section: Home, New Chat, Notifications, Agent Marketplace ──
    // Visual order: Header → Home → New Chat → Search → Notifications → Marketplace → ...
    // Search bar is React-managed (direct child of outerWrapper), so we use CSS order
    // to visually position it after Home + New Chat
    var staticSection = document.getElementById(STATIC_ID);
    if (!staticSection) {
      var NEW_CHAT_SVG = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M16.7929 2.79289C18.0118 1.57394 19.9882 1.57394 21.2071 2.79289C22.4261 4.01184 22.4261 5.98815 21.2071 7.20711L12.7071 15.7071C12.5196 15.8946 12.2652 16 12 16H9C8.44772 16 8 15.5523 8 15V12C8 11.7348 8.10536 11.4804 8.29289 11.2929L16.7929 2.79289ZM19.7929 4.20711C19.355 3.7692 18.645 3.7692 18.2071 4.2071L10 12.4142V14H11.5858L19.7929 5.79289C20.2308 5.35499 20.2308 4.64501 19.7929 4.20711ZM6 5C5.44772 5 5 5.44771 5 6V18C5 18.5523 5.44772 19 6 19H18C18.5523 19 19 18.5523 19 18V14C19 13.4477 19.4477 13 20 13C20.5523 13 21 13.4477 21 14V18C21 19.6569 19.6569 21 18 21H6C4.34315 21 3 19.6569 3 18V6C3 4.34314 4.34315 3 6 3H10C10.5523 3 11 3.44771 11 4C11 4.55228 10.5523 5 10 5H6Z" fill="currentColor"></path></svg>';
      staticSection = document.createElement('div');
      staticSection.id = STATIC_ID;
      staticSection.style.cssText = 'width:100%;flex-shrink:0;box-sizing:border-box;padding:0 0.25rem;order:2;';
      staticSection.appendChild(createSidebarBtn('sv-sb-home', 'Home', homeIcon, SB_BASE + '/home'));
      staticSection.appendChild(createSidebarBtn('sv-new-chat-btn', 'New chat', NEW_CHAT_SVG, '/c/new'));
      staticSection.appendChild(createSidebarBtn('sv-sb-notifications', 'Notifications', notifIcon, SB_BASE + '/notifications'));
      staticSection.appendChild(createSidebarBtn(MKT_ID, 'Agent Marketplace', MKT_SVG, '/agents'));
      // scrollContainer must be a child of outerWrapper for insertBefore to work
      if (scrollContainer.parentElement === outerWrapper) {
        outerWrapper.insertBefore(staticSection, scrollContainer);
      } else {
        outerWrapper.appendChild(staticSection);
      }
    }

    // Apply CSS order to outerWrapper children for visual ordering
    // Header=1, Static(Home,NewChat)=2, SearchBar=3, NavWrap=4, ScrollContainer=5
    if (svHeader) svHeader.style.order = '1';
    // Find and tag the React search bar
    var searchBar = outerWrapper.querySelector('.flex.items-center.justify-between.px-0\\.5');
    if (!searchBar) {
      // Try finding by the search input container
      for (var si = 0; si < outerWrapper.children.length; si++) {
        var oc = outerWrapper.children[si];
        if (oc.querySelector && oc.querySelector('input[placeholder]') && !oc.id) {
          searchBar = oc; break;
        }
      }
    }
    if (searchBar) searchBar.style.order = '3';

    // Move Automations into static section (after Agent Marketplace)
    // IG Template is now inside the Automations panel — hide its separate button
    var autoBtn = document.getElementById('nanobot-automations-btn');
    var tmplBtn = document.getElementById('sv-template-btn');
    if (autoBtn && autoBtn.parentElement !== staticSection) {
      staticSection.appendChild(autoBtn);
    }
    if (tmplBtn) {
      tmplBtn.style.display = 'none';
    }

    // ── SCROLLABLE wrap: Mission Control, Social, Street items, + Chats ──
    var wrap = document.getElementById(WRAP_ID);
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.id = WRAP_ID;
      wrap.style.cssText = 'width:100%;flex-shrink:1;box-sizing:border-box;padding:0 0.25rem;order:4;';

      // Social button removed — Messages now links to /social/dm

      var SB_ITEMS = [
        { id: 'sv-sb-profile', label: 'Street Profile', svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>', path: SB_BASE + '/profile' },
        { id: 'sv-sb-forum', label: 'Word On The Street', icon: '/images/sidebar-icons/word.svg', path: SB_BASE + '/forum' },
        { id: 'sv-sb-gallery', label: 'Street Gallery', icon: '/images/sidebar-icons/gallery.svg', path: SB_BASE + '/gallery' },
        { id: 'sv-sb-groups', label: 'Groups', svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>', path: SB_BASE + '/groups' },
        { id: 'sv-sb-news', label: 'News', icon: '/images/sidebar-icons/news.svg', path: SB_BASE + '/news' },
        { id: 'sv-sb-messages', label: 'Messages', icon: '/images/sidebar-icons/messages-bubble.svg', path: '/social/dm' },
        { id: 'sv-sb-directory', label: 'Directory', icon: '/images/sidebar-icons/directory-grid.svg', path: SB_BASE + '/directory' },
        { id: 'sv-sb-jobs', label: 'Job Board', icon: '/images/sidebar-icons/job-briefcase.svg', path: SB_BASE + '/jobs' },
        { id: 'sv-sb-academy', label: 'Academy', icon: '/images/sidebar-icons/lms-cap.svg', path: SB_BASE + '/academy' },
        { id: 'sv-sb-calendar', label: 'Calendar', icon: '/images/sidebar-icons/calendar-square.svg', path: SB_BASE + '/calendar' },
        { id: 'sv-sb-social', label: 'Social Media', icon: '/images/sidebar-icons/social-media.svg', path: SB_BASE + '/social-media' },
        { id: 'sv-sb-tasks', label: 'Tasks', icon: '/images/sidebar-icons/tasks-clipboard.svg', path: SB_BASE + '/tasks' },
        { id: 'sv-sb-documents', label: 'Documents', icon: '/images/sidebar-icons/documents.svg', path: SB_BASE + '/documents' },
        { id: 'sv-sb-storage', label: 'Storage', icon: '/images/sidebar-icons/storage.svg', path: SB_BASE + '/storage' },
        { id: 'sv-sb-database', label: 'Database', icon: '/images/sidebar-icons/database-grid.svg', path: SB_BASE + '/data' },
        { id: 'sv-sb-grantwriter', label: 'Grant Writer', icon: '/images/sidebar-icons/grantwriter.svg', path: SB_BASE + '/grantwriter' },
        { id: 'sv-sb-accounting', label: 'Accounting', icon: '/images/sidebar-icons/accounting.svg', path: SB_BASE + '/accounting' },
      ];
      SB_ITEMS.forEach(function (item) {
        var iconHtml = item.svg ? item.svg : '<img src="' + item.icon + '" alt="" width="20" height="20" class="sv-sidebar-icon" style="opacity:0.7;flex-shrink:0;" />';
        wrap.appendChild(createSidebarBtn(item.id, item.label, iconHtml, item.path));
      });

      // scrollContainer must be a child of outerWrapper for insertBefore to work
      if (scrollContainer.parentElement === outerWrapper) {
        outerWrapper.insertBefore(wrap, scrollContainer);
      } else {
        outerWrapper.appendChild(wrap);
      }
    }

    // Style scrollContainer in place — do NOT reparent it (breaks React reconciliation)
    scrollContainer.style.order = '5';
    scrollContainer.style.flex = '1 1 auto';
    scrollContainer.style.minHeight = '200px';
    scrollContainer.style.overflow = 'hidden';
    // Let ReactVirtualized's list scroll, with scrollbar visible on hover
    if (!document.getElementById('sv-hide-rv-scrollbar')) {
      var rvStyle = document.createElement('style');
      rvStyle.id = 'sv-hide-rv-scrollbar';
      rvStyle.textContent = [
        '.ReactVirtualized__List { overflow-y: auto !important; scrollbar-width: thin !important; scrollbar-color: transparent transparent !important; }',
        'nav#chat-history-nav:hover .ReactVirtualized__List { scrollbar-color: rgba(128,128,128,0.3) transparent !important; }',
        '.ReactVirtualized__List::-webkit-scrollbar { width: 4px !important; background: transparent !important; }',
        '.ReactVirtualized__List::-webkit-scrollbar-track { background: transparent !important; }',
        '.ReactVirtualized__List::-webkit-scrollbar-thumb { background: transparent !important; border-radius: 4px !important; }',
        '.dark nav#chat-history-nav:hover .ReactVirtualized__List::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15) !important; }',
        'html:not(.dark) nav#chat-history-nav:hover .ReactVirtualized__List::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15) !important; }',
      ].join('\n');
      document.head.appendChild(rvStyle);
    }
  }

  setInterval(injectSidebarButtons, 600);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { setTimeout(injectSidebarButtons, 500); });
  } else { setTimeout(injectSidebarButtons, 500); }

  // ── Standalone sidebar for non-LibreChat pages (Social, LobeHub) ──
  var STANDALONE_ID = 'sv-standalone-sidebar';
  var standaloneSidebarCollapsed = false;

  function isNonLibreChatPage() {
    var p = window.location.pathname;
    return p.startsWith('/social') || p.startsWith('/discover') || p.startsWith('/accounting');
  }

  function injectStandaloneSidebar() {
    if (!isNonLibreChatPage()) return;
    // Don't inject when inside an iframe (e.g., embedded in Street Profile)
    if (window !== window.top) return;
    if (document.getElementById(STANDALONE_ID)) return;
    // Don't inject if LibreChat's sidebar already exists
    if (document.getElementById('chat-history-nav')) return;

    var isDark = document.documentElement.classList.contains('dark') ||
                 document.body.classList.contains('dark') ||
                 (window.localStorage && localStorage.getItem('theme') === 'dark');

    // Inject standalone sidebar CSS
    if (!document.getElementById('sv-standalone-css')) {
      var css = document.createElement('style');
      css.id = 'sv-standalone-css';
      css.textContent = [
        '#' + STANDALONE_ID + ' { position: fixed; left: 0; top: 0; bottom: 0; width: 260px; z-index: 110; display: flex; flex-direction: column; transition: width 0.2s ease; overflow: hidden; }',
        '#' + STANDALONE_ID + '.sv-collapsed { width: 56px; }',
        '.dark #' + STANDALONE_ID + ' { background: var(--surface-primary-alt, #171717); border-right: 1px solid rgba(255,255,255,0.08); }',
        'html:not(.dark) #' + STANDALONE_ID + ' { background: var(--surface-primary-alt, #f9f9f9); border-right: 1px solid rgba(0,0,0,0.08); }',
        // Hide text in collapsed mode
        '#' + STANDALONE_ID + '.sv-collapsed [role="button"] span { display: none !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed [role="button"] > div { padding-right: 0 !important; justify-content: center !important; flex: unset !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed [role="button"] > div > div { margin-right: 0 !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed [role="button"] { justify-content: center !important; padding: 8px 0 !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed .sv-sa-header { padding: 8px 0 !important; justify-content: center !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed .sv-sa-header img[width="130"] { display: none !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed .sv-sa-header button { display: none !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed .sv-sa-header { cursor: pointer; }',
        '#' + STANDALONE_ID + '.sv-collapsed .sv-sa-nav-wrap { padding: 0 !important; scrollbar-width: none !important; }',
        '#' + STANDALONE_ID + '.sv-collapsed .sv-sa-static { padding: 0 !important; }',
        // Dark mode icon fix
        '.dark #' + STANDALONE_ID + ' svg { color: white !important; }',
        '.dark #' + STANDALONE_ID + ' .sv-sidebar-icon { filter: brightness(0) invert(1) !important; opacity: 1 !important; }',
        'html:not(.dark) #' + STANDALONE_ID + ' .sv-sidebar-icon { filter: brightness(0) !important; }',
        'html:not(.dark) #' + STANDALONE_ID + ' .sv-logo-img { filter: brightness(0) !important; }',
        '.dark #' + STANDALONE_ID + ' .sv-logo-img { filter: none !important; }',
        // Shift Social app content
        '.sv-has-standalone-sidebar { margin-left: 260px !important; transition: margin-left 0.2s ease !important; }',
        '.sv-has-standalone-sidebar-collapsed { margin-left: 56px !important; transition: margin-left 0.2s ease !important; }',
        // Hide Social app\'s own sidebar when standalone sidebar is present
        'body:has(#' + STANDALONE_ID + ') aside.w-64 { display: none !important; }',
        'body:has(#' + STANDALONE_ID + ') .flex.h-screen.overflow-hidden { margin-left: 260px; width: calc(100vw - 260px); transition: margin-left 0.2s ease, width 0.2s ease; }',
        'body:has(#' + STANDALONE_ID + '.sv-collapsed) .flex.h-screen.overflow-hidden { margin-left: 56px; width: calc(100vw - 56px); }',
        // Scrollbar on the nav wrap
        '#' + STANDALONE_ID + ' .sv-sa-nav-wrap { overflow-y: auto; scrollbar-width: thin; scrollbar-color: transparent transparent; }',
        '#' + STANDALONE_ID + ':hover .sv-sa-nav-wrap { scrollbar-color: rgba(128,128,128,0.3) transparent; }',
        '#' + STANDALONE_ID + ' .sv-sa-nav-wrap::-webkit-scrollbar { width: 4px; background: transparent; }',
        '#' + STANDALONE_ID + ' .sv-sa-nav-wrap::-webkit-scrollbar-thumb { background: transparent; border-radius: 4px; }',
        '#' + STANDALONE_ID + ':hover .sv-sa-nav-wrap::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.3); }',
      ].join('\n');
      document.head.appendChild(css);
    }

    // Create sidebar container
    var sidebar = document.createElement('div');
    sidebar.id = STANDALONE_ID;

    // Header with SV logo + collapse button
    var header = document.createElement('div');
    header.className = 'sv-sa-header';
    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:0.5rem 1.25rem 0.5rem 0.75rem;width:100%;flex-shrink:0;';
    var logoWrap = document.createElement('div');
    logoWrap.style.cssText = 'display:flex;align-items:center;gap:10px;cursor:pointer;margin-top:4px;';
    logoWrap.innerHTML = '<img class="sv-logo-img" src="/images/streetbot/megaphone-icon.svg" alt="Street Voices" width="32" height="32" style="width:32px;height:32px;min-width:28px;flex-shrink:0;" /><img class="sv-logo-img" src="/images/streetbot/streetvoices-text.svg" alt="Street Voices" width="130" height="19" style="max-width:100%;height:auto;" />';
    logoWrap.addEventListener('click', function() { window.location.href = '/'; });
    header.appendChild(logoWrap);

    var collapseBtn = document.createElement('button');
    collapseBtn.style.cssText = 'cursor:pointer;display:flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:0.5rem;border:none;background:transparent;color:var(--text-primary);margin-left:auto;flex-shrink:0;';
    collapseBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>';
    collapseBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      standaloneSidebarCollapsed = !standaloneSidebarCollapsed;
      applyStandaloneState();
    });
    header.appendChild(collapseBtn);

    // Make header expand when collapsed
    header.addEventListener('click', function(e) {
      if (!standaloneSidebarCollapsed) return;
      e.stopPropagation();
      standaloneSidebarCollapsed = false;
      applyStandaloneState();
    });

    sidebar.appendChild(header);

    // Static section: Home, New Chat, Notifications, Agent Marketplace, Automations, IG Template
    var staticSection = document.createElement('div');
    staticSection.className = 'sv-sa-static';
    staticSection.style.cssText = 'width:100%;flex-shrink:0;box-sizing:border-box;padding:0 0.25rem;';

    var homeIcon = '<img src="/images/sidebar-icons/home.svg" alt="" width="20" height="20" class="sv-sidebar-icon" style="opacity:0.7;flex-shrink:0;" />';
    var notifIcon = '<img src="/images/sidebar-icons/notifications.svg" alt="" width="20" height="20" class="sv-sidebar-icon" style="opacity:0.7;flex-shrink:0;" />';
    var NEW_CHAT_SVG = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M16.7929 2.79289C18.0118 1.57394 19.9882 1.57394 21.2071 2.79289C22.4261 4.01184 22.4261 5.98815 21.2071 7.20711L12.7071 15.7071C12.5196 15.8946 12.2652 16 12 16H9C8.44772 16 8 15.5523 8 15V12C8 11.7348 8.10536 11.4804 8.29289 11.2929L16.7929 2.79289ZM19.7929 4.20711C19.355 3.7692 18.645 3.7692 18.2071 4.2071L10 12.4142V14H11.5858L19.7929 5.79289C20.2308 5.35499 20.2308 4.64501 19.7929 4.20711ZM6 5C5.44772 5 5 5.44771 5 6V18C5 18.5523 5.44772 19 6 19H18C18.5523 19 19 18.5523 19 18V14C19 13.4477 19.4477 13 20 13C20.5523 13 21 13.4477 21 14V18C21 19.6569 19.6569 21 18 21H6C4.34315 21 3 19.6569 3 18V6C3 4.34314 4.34315 3 6 3H10C10.5523 3 11 3.44771 11 4C11 4.55228 10.5523 5 10 5H6Z" fill="currentColor"></path></svg>';
    var MKT_SVG2 = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>';

    staticSection.appendChild(createSidebarBtn('sv-sa-home', 'Home', homeIcon, '/home'));
    staticSection.appendChild(createSidebarBtn('sv-sa-newchat', 'New chat', NEW_CHAT_SVG, '/c/new'));
    staticSection.appendChild(createSidebarBtn('sv-sa-notif', 'Notifications', notifIcon, '/notifications'));
    staticSection.appendChild(createSidebarBtn('sv-sa-mkt', 'Agent Marketplace', MKT_SVG2, '/agents'));
    sidebar.appendChild(staticSection);

    // Nav wrap: all the menu items
    var navWrap = document.createElement('div');
    navWrap.className = 'sv-sa-nav-wrap';
    navWrap.style.cssText = 'width:100%;flex:1 1 auto;box-sizing:border-box;padding:0 0.25rem;overflow-y:auto;';

    var SB_ITEMS = [
      { id: 'sv-sa-profile', label: 'Street Profile', svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>', path: '/profile' },
      { id: 'sv-sa-forum', label: 'Word On The Street', icon: '/images/sidebar-icons/word.svg', path: '/forum' },
      { id: 'sv-sa-gallery', label: 'Street Gallery', icon: '/images/sidebar-icons/gallery.svg', path: '/gallery' },
      { id: 'sv-sa-groups', label: 'Groups', svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>', path: '/groups' },
      { id: 'sv-sa-news', label: 'News', icon: '/images/sidebar-icons/news.svg', path: '/news' },
      { id: 'sv-sa-messages', label: 'Messages', icon: '/images/sidebar-icons/messages-bubble.svg', path: '/social/dm' },
      { id: 'sv-sa-directory', label: 'Directory', icon: '/images/sidebar-icons/directory-grid.svg', path: '/directory' },
      { id: 'sv-sa-jobs', label: 'Job Board', icon: '/images/sidebar-icons/job-briefcase.svg', path: '/jobs' },
      { id: 'sv-sa-academy', label: 'Academy', icon: '/images/sidebar-icons/lms-cap.svg', path: '/academy' },
      { id: 'sv-sa-calendar', label: 'Calendar', icon: '/images/sidebar-icons/calendar-square.svg', path: '/calendar' },
      { id: 'sv-sa-social', label: 'Social Media', icon: '/images/sidebar-icons/social-media.svg', path: '/social-media' },
      { id: 'sv-sa-tasks', label: 'Tasks', icon: '/images/sidebar-icons/tasks-clipboard.svg', path: '/tasks' },
      { id: 'sv-sa-documents', label: 'Documents', icon: '/images/sidebar-icons/documents.svg', path: '/documents' },
      { id: 'sv-sa-storage', label: 'Storage', icon: '/images/sidebar-icons/storage.svg', path: '/storage' },
      { id: 'sv-sa-database', label: 'Database', icon: '/images/sidebar-icons/database-grid.svg', path: '/data' },
      { id: 'sv-sa-grantwriter', label: 'Grant Writer', icon: '/images/sidebar-icons/grantwriter.svg', path: '/grantwriter' },
      { id: 'sv-sa-accounting', label: 'Accounting', icon: '/images/sidebar-icons/accounting.svg', path: '/accounting' },
    ];
    SB_ITEMS.forEach(function (item) {
      var iconHtml = item.svg ? item.svg : '<img src="' + item.icon + '" alt="" width="20" height="20" class="sv-sidebar-icon" style="opacity:0.7;flex-shrink:0;" />';
      navWrap.appendChild(createSidebarBtn(item.id, item.label, iconHtml, item.path));
    });
    sidebar.appendChild(navWrap);

    document.body.appendChild(sidebar);

    // Apply theme from localStorage
    if (isDark && !document.documentElement.classList.contains('dark')) {
      document.documentElement.classList.add('dark');
    }

    // Hide Social app's own sidebar and shift content
    applyStandaloneState();
  }

  function applyStandaloneState() {
    var sidebar = document.getElementById(STANDALONE_ID);
    if (!sidebar) return;

    if (standaloneSidebarCollapsed) {
      sidebar.classList.add('sv-collapsed');
    } else {
      sidebar.classList.remove('sv-collapsed');
    }

    // Find the app's root content and shift it
    // Social app: div.flex.h-screen.overflow-hidden
    var socialRoot = document.querySelector('.flex.h-screen.overflow-hidden');
    if (socialRoot) {
      // Hide Social's internal sidebar (first child if it's the sidebar component)
      var socialSidebar = socialRoot.querySelector('aside') || socialRoot.querySelector('nav:first-child');
      // The Social sidebar is the first direct child that's not <main>
      for (var i = 0; i < socialRoot.children.length; i++) {
        var child = socialRoot.children[i];
        if (child.tagName !== 'MAIN' && child.id !== STANDALONE_ID) {
          // Check if it looks like a sidebar (narrow, fixed width)
          if (child.tagName === 'ASIDE' || child.tagName === 'NAV' ||
              (child.classList.contains('w-64') || child.classList.contains('w-60') ||
               child.querySelector('[class*="sidebar"]') || child.querySelector('nav'))) {
            child.style.display = 'none';
          }
        }
      }
      socialRoot.style.marginLeft = standaloneSidebarCollapsed ? '56px' : '260px';
      socialRoot.style.transition = 'margin-left 0.2s ease';
      socialRoot.style.width = standaloneSidebarCollapsed ? 'calc(100% - 56px)' : 'calc(100% - 260px)';
    }

    // LobeHub: try to find its main container
    var lobeRoot = document.querySelector('#__next') || document.querySelector('main') || document.body.firstElementChild;
    if (lobeRoot && !socialRoot && lobeRoot.id !== STANDALONE_ID) {
      lobeRoot.style.marginLeft = standaloneSidebarCollapsed ? '56px' : '260px';
      lobeRoot.style.transition = 'margin-left 0.2s ease';
    }
  }

  // Poll for standalone sidebar injection
  if (isNonLibreChatPage()) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () { setTimeout(injectStandaloneSidebar, 300); });
    } else { setTimeout(injectStandaloneSidebar, 300); }
    setInterval(function () {
      if (!document.getElementById(STANDALONE_ID)) injectStandaloneSidebar();
      applyStandaloneState();
    }, 1000);
  }

  // ── ChatGPT-style collapse: icon-only sidebar when "closed" ──
  (function setupCollapse() {
    var CLS = 'sv-collapsed';
    // Inject collapse CSS once
    if (!document.getElementById('sv-collapse-css')) {
      var css = document.createElement('style');
      css.id = 'sv-collapse-css';
      css.textContent = [
        // Never hide the nav wrapper — override LibreChat translateX(-320px)
        '.nav.sv-collapsed { transform: none !important; width: 56px !important; transition: width 0.2s ease !important; position: fixed !important; left: 0 !important; top: 0 !important; z-index: 110 !important; opacity: 1 !important; pointer-events: auto !important; }',
        '.nav:not(.sv-collapsed) { transition: width 0.2s ease !important; opacity: 1 !important; }',
        // Hide all text labels in collapsed mode
        '.nav.sv-collapsed nav#chat-history-nav span { display: none !important; }',
        '.nav.sv-collapsed nav#chat-history-nav kbd { display: none !important; }',
        // SV header: hide text logo, center megaphone
        '.nav.sv-collapsed #sv-sidebar-header { padding: 8px 0 !important; justify-content: center !important; }',
        '.nav.sv-collapsed #sv-sidebar-header img[width="130"] { display: none !important; }',
        '.nav.sv-collapsed #sv-sidebar-header button { display: none !important; }',
        // Buttons: center icons, shrink padding
        '.nav.sv-collapsed nav#chat-history-nav [role="button"] { justify-content: center !important; padding: 8px 0 !important; }',
        '.nav.sv-collapsed nav#chat-history-nav [role="button"] > div { padding-right: 0 !important; justify-content: center !important; flex: unset !important; }',
        '.nav.sv-collapsed nav#chat-history-nav [role="button"] > div > div { margin-right: 0 !important; }',
        // Search bar: hide
        '.nav.sv-collapsed nav#chat-history-nav .px-3 { display: none !important; }',
        // Hide chats section
        '.nav.sv-collapsed .flex-grow.min-h-0 { display: none !important; }',
        '.nav.sv-collapsed h3 { display: none !important; }',
        // Static & wrap sections: no padding
        '.nav.sv-collapsed #sv-sidebar-static { padding: 0 !important; }',
        '.nav.sv-collapsed #sv-sidebar-nav-wrap { padding: 0 !important; overflow-y: auto !important; max-height: calc(100vh - 180px) !important; scrollbar-width: none !important; }',
        '.nav.sv-collapsed #sv-sidebar-nav-wrap::-webkit-scrollbar { width: 0 !important; }',
        // User footer: center avatar, hide name
        '.nav.sv-collapsed nav#chat-history-nav > div.flex-1 > div:last-child { padding: 0 !important; }',
        '.nav.sv-collapsed [data-testid="nav-user"] span { display: none !important; }',
        '.nav.sv-collapsed [data-testid="nav-user"] > div.flex-1 { display: none !important; }',
        '.nav.sv-collapsed [data-testid="nav-user"] { justify-content: center !important; }',
        // Nav itself
        '.nav.sv-collapsed nav#chat-history-nav { width: 56px !important; min-width: 56px !important; padding: 4px 0 !important; }',
        // Hide both native LibreChat sidebar buttons — we have our own in the SV header
        'button[aria-label="Open sidebar"] { display: none !important; }',
        'button[aria-label="Close sidebar"] { display: none !important; }',
        // When collapsed, make header look clickable
        '.nav.sv-collapsed #sv-sidebar-header { cursor: pointer; }',
        // Shift main content right when collapsed (JS handles this via marginLeft)
        // Tooltip on hover for collapsed icons
        '.nav.sv-collapsed nav#chat-history-nav [role="button"]:hover { position: relative; }',
      ].join('\n');
      document.head.appendChild(css);
    }

    function getNavWrapper() {
      var nav = document.getElementById('chat-history-nav');
      if (!nav) return null;
      var p = nav.parentElement;
      if (p) p = p.parentElement; // grandparent = .nav div
      if (p && p.classList.contains('nav')) return p;
      return null;
    }

    var collapsed = false;
    var expandLock = false; // Prevent watchTransform from re-collapsing during expand

    // Listen for collapse events from the SV header button
    document.addEventListener('sv-collapse-sidebar', function() {
      collapsed = true;
      applyState();
    });

    function applyState() {
      var wrapper = getNavWrapper();
      if (!wrapper) return;

      // The flex-shrink-0 parent is the flex slot LibreChat allocates for the sidebar
      var flexSlot = wrapper.parentElement;
      var rootFlex = document.querySelector('#root > div.flex');

      if (collapsed) {
        wrapper.classList.add(CLS);
        wrapper.style.transform = 'none';
        wrapper.style.width = '56px';
        // Collapse the flex slot to 0 so content fills full width
        if (flexSlot && flexSlot.classList.contains('flex-shrink-0')) {
          flexSlot.style.width = '0px';
          flexSlot.style.minWidth = '0px';
        }
        if (rootFlex) rootFlex.style.marginLeft = '56px';
      } else {
        wrapper.classList.remove(CLS);
        wrapper.style.width = '';
        wrapper.style.transform = '';
        wrapper.style.opacity = '1';
        wrapper.style.pointerEvents = '';
        wrapper.style.position = '';
        wrapper.style.left = '';
        wrapper.style.top = '';
        wrapper.style.zIndex = '';
        // Restore the flex slot
        if (flexSlot && flexSlot.classList.contains('flex-shrink-0')) {
          flexSlot.style.width = '';
          flexSlot.style.minWidth = '';
        }
        if (rootFlex) rootFlex.style.marginLeft = '';
      }
    }

    // Watch for LibreChat trying to set translateX(-320px) and intercept
    function watchTransform() {
      if (expandLock) return; // Don't interfere during expand animation
      var wrapper = getNavWrapper();
      if (!wrapper) return;

      // Check if LibreChat set a negative translateX (meaning it wants to close)
      var currentTransform = wrapper.style.transform;
      if (currentTransform && currentTransform.indexOf('translateX(-') >= 0) {
        // LibreChat wants to close — switch to collapsed mode
        collapsed = true;
        applyState();
      } else if (currentTransform === 'none' || currentTransform === '') {
        // Check if it's supposed to be open (width 320px)
        if (wrapper.classList.contains(CLS) && !collapsed) {
          wrapper.classList.remove(CLS);
        }
      }
    }

    // No longer needed — native close button hidden via CSS, custom event handles collapse

    // Make the SV header logo expand the sidebar when clicked in collapsed mode
    function ensureHeaderExpand() {
      var svHeader = document.getElementById('sv-sidebar-header');
      if (!svHeader || svHeader.dataset.svExpandBound) return;
      svHeader.dataset.svExpandBound = 'true';
      svHeader.addEventListener('click', function(e) {
        if (!collapsed) return; // Only expand when collapsed
        e.stopPropagation();
        e.preventDefault();
        collapsed = false;
        expandLock = true;
        applyState();
        // Trigger LibreChat's own "Open sidebar" to sync its internal state
        var openBtn = document.querySelector('button[aria-label="Open sidebar"]');
        if (openBtn) openBtn.click();
        setTimeout(function() { expandLock = false; }, 1000);
      });
    }

    // Poll to enforce state
    setInterval(function() {
      watchTransform();
      ensureHeaderExpand();

      // Keep enforcing transform override when collapsed
      if (collapsed && !expandLock) {
        var wrapper = getNavWrapper();
        if (wrapper) {
          if (wrapper.style.transform !== 'none') {
            wrapper.style.transform = 'none';
          }
          if (!wrapper.classList.contains(CLS)) {
            wrapper.classList.add(CLS);
          }
          wrapper.style.width = '56px';
        }
      }
    }, 300);
  })();

  // /agents route now handled natively by React router — no redirect needed
})();
