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
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
