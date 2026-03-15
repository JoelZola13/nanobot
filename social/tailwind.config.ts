import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#0A0A0A",
          surface: "#141414",
          elevated: "#1C1C1C",
          hover: "#262626",
        },
        border: {
          DEFAULT: "#262626",
          subtle: "#1C1C1C",
          focus: "#FF6B35",
        },
        accent: {
          DEFAULT: "#FF6B35",
          hover: "#FF8A5C",
          muted: "rgba(255, 107, 53, 0.15)",
        },
        teal: {
          DEFAULT: "#00D4AA",
          hover: "#00E8BC",
          muted: "rgba(0, 212, 170, 0.15)",
        },
        text: {
          primary: "#F5F5F5",
          secondary: "#A3A3A3",
          muted: "#666666",
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
    },
  },
  plugins: [],
};
export default config;
