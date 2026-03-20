import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "var(--sv-bg)",
          surface: "var(--sv-bg-surface)",
          elevated: "var(--sv-bg-elevated)",
          hover: "var(--sv-bg-hover)",
        },
        border: {
          DEFAULT: "var(--sv-border)",
          subtle: "var(--sv-border-subtle)",
          focus: "var(--sv-accent)",
        },
        accent: {
          DEFAULT: "var(--sv-accent)",
          hover: "var(--sv-accent-hover)",
          muted: "var(--sv-accent-muted)",
        },
        teal: {
          DEFAULT: "#00D4AA",
          hover: "#00E8BC",
          muted: "rgba(0, 212, 170, 0.15)",
        },
        text: {
          primary: "var(--sv-text-primary)",
          secondary: "var(--sv-text-secondary)",
          muted: "var(--sv-text-muted)",
        },
        danger: {
          DEFAULT: "#EF4444",
          muted: "rgba(239, 68, 68, 0.15)",
        },
      },
      fontFamily: {
        heading: ['"Space Grotesk"', "system-ui", "sans-serif"],
        body: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "monospace"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      backdropBlur: {
        glass: "16px",
      },
    },
  },
  plugins: [],
};
export default config;
