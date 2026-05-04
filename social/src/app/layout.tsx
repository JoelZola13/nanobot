import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Street Voices Social",
  description: "Where humans and AI collaborate",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            try {
              var path = window.location.pathname;
              var isDmPage = path === '/social/dm' || path.indexOf('/social/dm/') === 0;
              var isEmbedded = window.self !== window.top;
              var params = new URLSearchParams(window.location.search);
              if (isDmPage && !isEmbedded && params.get('embed') !== 'true') {
                window.location.replace('/messages');
              }
            } catch(e) {}
          })();
        `}} />
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            try {
              var isEmbedded = window.self !== window.top || new URLSearchParams(window.location.search).get('embed') === 'true';
              if (isEmbedded) return;
              var script = document.createElement('script');
              script.src = 'http://localhost:18790/shared/sv-platform.js?v=89';
              script.defer = true;
              document.head.appendChild(script);
            } catch(e) {}
          })();
        `}} />
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            function normalizeTheme(value) {
              return value === 'light' || value === 'dark' ? value : null;
            }

            function readParentTheme() {
              try {
                if (window.self === window.top || !window.parent) return null;
                var parentHtml = window.parent.document.documentElement;
                var parentTheme = normalizeTheme(parentHtml.getAttribute('data-theme'));
                if (parentTheme) return parentTheme;
                if (parentHtml.classList.contains('light')) return 'light';
                if (parentHtml.classList.contains('dark')) return 'dark';
                return normalizeTheme(window.parent.localStorage.getItem('theme')) ||
                  normalizeTheme(window.parent.localStorage.getItem('color-theme'));
              } catch(e) {
                return null;
              }
            }

            function resolveTheme() {
              var params = new URLSearchParams(window.location.search);
              return normalizeTheme(params.get('theme')) ||
                readParentTheme() ||
                normalizeTheme(localStorage.getItem('theme')) ||
                normalizeTheme(localStorage.getItem('color-theme')) ||
                (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
            }

            function setTheme(theme) {
              var nextTheme = normalizeTheme(theme) || resolveTheme();
              document.documentElement.classList.remove('dark', 'light');
              document.documentElement.classList.add(nextTheme);
              document.documentElement.setAttribute('data-theme', nextTheme);
              document.documentElement.style.colorScheme = nextTheme;
            }

            function applyTheme() {
              try {
                setTheme(resolveTheme());
              } catch(e) {}
            }

            applyTheme();
            window.addEventListener('message', function(e) {
              if (e.origin !== window.location.origin) return;
              var data = e.data || {};
              if (data.type === 'street-voices-theme') {
                setTheme(data.theme);
              }
            });
            // Re-check theme when tab becomes visible (user may have toggled on home page)
            document.addEventListener('visibilitychange', function() {
              if (!document.hidden) applyTheme();
            });
            // Listen for storage changes from other tabs/pages
            window.addEventListener('storage', function(e) {
              if (e.key === 'theme' || e.key === 'color-theme') applyTheme();
            });
            // Poll every 500ms in case same-tab navigation changed it
            setInterval(applyTheme, 500);
          })();
        `}} />
      </head>
      <body>{children}</body>
    </html>
  );
}
