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
        <script src="http://localhost:18790/shared/sv-platform.js" defer />
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            function applyTheme() {
              try {
                var theme = localStorage.getItem('color-theme');
                if (theme === 'light') {
                  document.documentElement.classList.remove('dark');
                } else {
                  document.documentElement.classList.add('dark');
                }
              } catch(e) {}
            }
            applyTheme();
            // Re-check theme when tab becomes visible (user may have toggled on home page)
            document.addEventListener('visibilitychange', function() {
              if (!document.hidden) applyTheme();
            });
            // Listen for storage changes from other tabs/pages
            window.addEventListener('storage', function(e) {
              if (e.key === 'color-theme') applyTheme();
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
