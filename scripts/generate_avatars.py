#!/usr/bin/env python3
"""
Generate unique SVG avatars for all 38 nanobot agents.

Each agent gets a team-colored avatar with role-based design differentiation:
- Lead agents: gradient background + crown glyph + bold initials
- Memory agents: gradient background + brain glyph + initials
- Member agents: gradient background + clean initials

Run: python scripts/generate_avatars.py
Output: static/avatars/{agent_name}.svg (38 files)
"""

import os
from pathlib import Path

# ── Team Color Palette ──────────────────────────────────────────────────────
TEAM_COLORS = {
    "executive":     {"primary": "#D4AF37", "accent": "#1a1a2e", "light": "#f5e6b8"},
    "communication": {"primary": "#3B82F6", "accent": "#1e3a5f", "light": "#bfdbfe"},
    "content":       {"primary": "#10B981", "accent": "#064e3b", "light": "#a7f3d0"},
    "development":   {"primary": "#8B5CF6", "accent": "#3b0764", "light": "#ddd6fe"},
    "finance":       {"primary": "#F59E0B", "accent": "#78350f", "light": "#fde68a"},
    "grant_writing": {"primary": "#14B8A6", "accent": "#134e4a", "light": "#99f6e4"},
    "research":      {"primary": "#6366F1", "accent": "#312e81", "light": "#c7d2fe"},
    "scraping":      {"primary": "#F97316", "accent": "#7c2d12", "light": "#fed7aa"},
}

# ── Agent Registry (all 38) ─────────────────────────────────────────────────
AGENTS = [
    # Executive
    {"name": "ceo",                      "team": "executive",     "role": "lead",   "initials": "CE", "display": "CEO"},
    {"name": "security_compliance",      "team": "executive",     "role": "member", "initials": "SC", "display": "Security"},
    {"name": "executive_memory",         "team": "executive",     "role": "memory", "initials": "EM", "display": "Exec Memory"},
    # Communication
    {"name": "communication_manager",    "team": "communication", "role": "lead",   "initials": "CM", "display": "Comms Lead"},
    {"name": "email_agent",              "team": "communication", "role": "member", "initials": "EA", "display": "Email"},
    {"name": "slack_agent",              "team": "communication", "role": "member", "initials": "SA", "display": "Slack"},
    {"name": "whatsapp_agent",           "team": "communication", "role": "member", "initials": "WA", "display": "WhatsApp"},
    {"name": "calendar_agent",           "team": "communication", "role": "member", "initials": "CA", "display": "Calendar"},
    {"name": "communication_memory",     "team": "communication", "role": "memory", "initials": "CM", "display": "Comms Memory"},
    # Content
    {"name": "content_manager",          "team": "content",       "role": "lead",   "initials": "CT", "display": "Content Lead"},
    {"name": "article_researcher",       "team": "content",       "role": "member", "initials": "AR", "display": "Researcher"},
    {"name": "article_writer",           "team": "content",       "role": "member", "initials": "AW", "display": "Writer"},
    {"name": "social_media_manager",     "team": "content",       "role": "member", "initials": "SM", "display": "Social"},
    {"name": "content_memory",           "team": "content",       "role": "memory", "initials": "CM", "display": "Content Memory"},
    # Development
    {"name": "development_manager",      "team": "development",   "role": "lead",   "initials": "DM", "display": "Dev Lead"},
    {"name": "backend_developer",        "team": "development",   "role": "member", "initials": "BE", "display": "Backend"},
    {"name": "frontend_developer",       "team": "development",   "role": "member", "initials": "FE", "display": "Frontend"},
    {"name": "database_manager",         "team": "development",   "role": "member", "initials": "DB", "display": "Database"},
    {"name": "devops",                   "team": "development",   "role": "member", "initials": "DO", "display": "DevOps"},
    {"name": "development_memory",       "team": "development",   "role": "memory", "initials": "DM", "display": "Dev Memory"},
    # Finance
    {"name": "finance_manager",          "team": "finance",       "role": "lead",   "initials": "FM", "display": "Finance Lead"},
    {"name": "accounting_agent",         "team": "finance",       "role": "member", "initials": "AC", "display": "Accounting"},
    {"name": "crypto_agent",             "team": "finance",       "role": "member", "initials": "CR", "display": "Crypto"},
    {"name": "finance_memory",           "team": "finance",       "role": "memory", "initials": "FM", "display": "Finance Memory"},
    # Grant Writing
    {"name": "grant_manager",            "team": "grant_writing", "role": "lead",   "initials": "GM", "display": "Grant Lead"},
    {"name": "grant_writer",             "team": "grant_writing", "role": "member", "initials": "GW", "display": "Grant Writer"},
    {"name": "budget_manager",           "team": "grant_writing", "role": "member", "initials": "BM", "display": "Budget"},
    {"name": "project_manager",          "team": "grant_writing", "role": "member", "initials": "PM", "display": "Projects"},
    {"name": "grant_memory",             "team": "grant_writing", "role": "memory", "initials": "GM", "display": "Grant Memory"},
    # Research
    {"name": "research_manager",         "team": "research",      "role": "lead",   "initials": "RM", "display": "Research Lead"},
    {"name": "media_platform_researcher","team": "research",      "role": "member", "initials": "MP", "display": "Platform"},
    {"name": "media_program_researcher", "team": "research",      "role": "member", "initials": "MR", "display": "Programs"},
    {"name": "street_bot_researcher",    "team": "research",      "role": "member", "initials": "SB", "display": "StreetBot"},
    {"name": "research_memory",          "team": "research",      "role": "memory", "initials": "RM", "display": "Research Memory"},
    # Scraping
    {"name": "scraping_manager",         "team": "scraping",      "role": "lead",   "initials": "SC", "display": "Scrape Lead"},
    {"name": "scraping_agent",           "team": "scraping",      "role": "member", "initials": "SA", "display": "Scraper"},
    {"name": "scraper_memory",           "team": "scraping",      "role": "memory", "initials": "SM", "display": "Scrape Memory"},
]


def _crown_glyph(cx: int, cy: int, color: str) -> str:
    """Small crown/star glyph for lead agents — positioned above initials."""
    y_top = cy - 28
    return f'''<g opacity="0.9">
    <polygon points="{cx-8},{y_top+10} {cx-5},{y_top+4} {cx},{y_top+7} {cx+5},{y_top+4} {cx+8},{y_top+10}"
             fill="{color}" stroke="none"/>
    <line x1="{cx-9}" y1="{y_top+10}" x2="{cx+9}" y2="{y_top+10}" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
  </g>'''


def _brain_glyph(cx: int, cy: int, color: str) -> str:
    """Small brain/circuit glyph for memory agents — positioned above initials."""
    y_top = cy - 28
    return f'''<g opacity="0.85" transform="translate({cx},{y_top+6})">
    <circle cx="0" cy="0" r="5" fill="none" stroke="{color}" stroke-width="1.2"/>
    <path d="M-3,-1 Q0,-4 3,-1" fill="none" stroke="{color}" stroke-width="1"/>
    <path d="M-3,1 Q0,4 3,1" fill="none" stroke="{color}" stroke-width="1"/>
    <line x1="0" y1="-5" x2="0" y2="-8" stroke="{color}" stroke-width="1"/>
    <line x1="-4" y1="-3" x2="-7" y2="-5" stroke="{color}" stroke-width="1"/>
    <line x1="4" y1="-3" x2="7" y2="-5" stroke="{color}" stroke-width="1"/>
    <circle cx="0" cy="-8" r="1.2" fill="{color}"/>
    <circle cx="-7" cy="-5" r="1.2" fill="{color}"/>
    <circle cx="7" cy="-5" r="1.2" fill="{color}"/>
  </g>'''


def _ring_decoration(cx: int, cy: int, r: int, color: str, role: str) -> str:
    """Outer decorative ring — thicker for leads, dashed for memory, subtle for members."""
    if role == "lead":
        return f'<circle cx="{cx}" cy="{cy}" r="{r+3}" fill="none" stroke="{color}" stroke-width="2" opacity="0.4"/>'
    elif role == "memory":
        return f'<circle cx="{cx}" cy="{cy}" r="{r+3}" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.35"/>'
    else:
        return f'<circle cx="{cx}" cy="{cy}" r="{r+2}" fill="none" stroke="{color}" stroke-width="1" opacity="0.2"/>'


def generate_svg(agent: dict) -> str:
    """Generate a complete SVG avatar for one agent."""
    colors = TEAM_COLORS[agent["team"]]
    primary = colors["primary"]
    accent = colors["accent"]
    light = colors["light"]
    name = agent["name"]
    initials = agent["initials"]
    role = agent["role"]

    # SVG dimensions
    size = 120
    cx, cy = size // 2, size // 2
    r = 46  # main circle radius

    # Gradient ID unique per agent
    grad_id = f"grad_{name}"
    bg_grad_id = f"bg_{name}"

    # Font weight varies by role
    font_weight = "700" if role == "lead" else "600" if role == "memory" else "500"
    font_size = "28" if role == "lead" else "26"

    # Build role-specific decorations
    glyph = ""
    if role == "lead":
        glyph = _crown_glyph(cx, cy, light)
    elif role == "memory":
        glyph = _brain_glyph(cx, cy, light)

    ring = _ring_decoration(cx, cy, r, primary, role)

    # Subtle inner shadow for depth
    shadow_id = f"shadow_{name}"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">
  <defs>
    <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{primary}"/>
      <stop offset="100%" stop-color="{accent}"/>
    </linearGradient>
    <radialGradient id="{bg_grad_id}" cx="35%" cy="35%" r="65%">
      <stop offset="0%" stop-color="{primary}" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
    </radialGradient>
    <filter id="{shadow_id}" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="{accent}" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- Outer ring decoration -->
  {ring}

  <!-- Main circle with gradient -->
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#{grad_id})" filter="url(#{shadow_id})"/>

  <!-- Inner highlight -->
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#{bg_grad_id})"/>

  <!-- Role glyph -->
  {glyph}

  <!-- Initials -->
  <text x="{cx}" y="{cy + 4 + (4 if role != 'member' else 0)}"
        text-anchor="middle" dominant-baseline="middle"
        font-family="'SF Pro Display', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
        font-size="{font_size}" font-weight="{font_weight}"
        fill="#ffffff" letter-spacing="1.5">
    {initials}
  </text>
</svg>'''

    return svg


def main():
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "static" / "avatars"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(AGENTS)} agent avatars...")

    for agent in AGENTS:
        svg_content = generate_svg(agent)
        filepath = output_dir / f"{agent['name']}.svg"
        filepath.write_text(svg_content)
        role_icon = {"lead": "★", "memory": "◎", "member": "●"}[agent["role"]]
        print(f"  {role_icon} {agent['name']:30s} [{agent['team']:15s}] → {filepath.name}")

    # Count by role
    leads = sum(1 for a in AGENTS if a["role"] == "lead")
    memories = sum(1 for a in AGENTS if a["role"] == "memory")
    members = sum(1 for a in AGENTS if a["role"] == "member")

    print(f"\n✓ Generated {len(AGENTS)} avatars: {leads} leads, {members} members, {memories} memory agents")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
