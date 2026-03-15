#!/usr/bin/env python3
"""Generate distinctive SVG avatars for all Nanobot agents.

Each avatar: 120x120 SVG with gradient circle + large centered icon glyph.
Designed to be recognizable at 40px (LobeHub card size).
No text labels — pure icon-based identification.
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# Team color palettes: (gradient_start, gradient_end, icon_color)
TEAM_COLORS = {
    "executive":     ("#6D28D9", "#1E1B4B", "#C4B5FD"),  # Purple/indigo
    "communication": ("#0EA5E9", "#0C4A6E", "#BAE6FD"),  # Sky blue
    "content":       ("#F59E0B", "#78350F", "#FDE68A"),  # Amber
    "development":   ("#10B981", "#064E3B", "#A7F3D0"),  # Emerald
    "finance":       ("#EF4444", "#7F1D1D", "#FECACA"),  # Red
    "grant_writing": ("#8B5CF6", "#3B0764", "#DDD6FE"),  # Violet
    "research":      ("#06B6D4", "#164E63", "#CFFAFE"),  # Cyan
    "scraping":      ("#F97316", "#7C2D12", "#FED7AA"),  # Orange
}

# Agent definitions: name -> (team, svg_icon_path)
# Icons are SVG path/shape snippets centered in the 120x120 viewBox
AGENTS = {
    # ═══ Executive ═══
    "auto": ("executive", """
    <!-- Lightning bolt -->
    <polygon points="68,28 52,58 64,58 52,88 76,48 62,48" fill="{ic}" stroke="none"/>
    """),

    "ceo": ("executive", """
    <!-- Crown -->
    <path d="M36,65 L42,42 L52,55 L60,35 L68,55 L78,42 L84,65 Z" fill="{ic}" stroke="none"/>
    <rect x="36" y="65" width="48" height="8" rx="2" fill="{ic}"/>
    """),

    "executive_memory": ("executive", """
    <!-- Brain -->
    <path d="M60,32 C48,32 40,40 40,50 C40,55 42,58 45,60 C43,63 42,67 44,72 C46,77 52,80 58,80 L60,80" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <path d="M60,32 C72,32 80,40 80,50 C80,55 78,58 75,60 C77,63 78,67 76,72 C74,77 68,80 62,80 L60,80" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="60" y1="38" x2="60" y2="74" stroke="{ic}" stroke-width="2.5"/>
    <line x1="48" y1="50" x2="60" y2="56" stroke="{ic}" stroke-width="2"/>
    <line x1="72" y1="50" x2="60" y2="56" stroke="{ic}" stroke-width="2"/>
    """),

    "security_compliance": ("executive", """
    <!-- Shield with checkmark -->
    <path d="M60,30 L82,42 L82,58 C82,72 72,82 60,88 C48,82 38,72 38,58 L38,42 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <polyline points="48,58 56,66 72,50" fill="none" stroke="{ic}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    """),

    # ═══ Communication ═══
    "communication_manager": ("communication", """
    <!-- Satellite/broadcast tower -->
    <circle cx="60" cy="42" r="5" fill="{ic}"/>
    <line x1="60" y1="47" x2="60" y2="80" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="48" y1="80" x2="72" y2="80" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <path d="M46,36 C46,28 54,22 60,22" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M74,36 C74,28 66,22 60,22" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M40,42 C38,30 48,18 60,18" fill="none" stroke="{ic}" stroke-width="2" stroke-linecap="round" opacity="0.6"/>
    <path d="M80,42 C82,30 72,18 60,18" fill="none" stroke="{ic}" stroke-width="2" stroke-linecap="round" opacity="0.6"/>
    """),

    "email_agent": ("communication", """
    <!-- Envelope -->
    <rect x="34" y="40" width="52" height="36" rx="4" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <polyline points="34,42 60,62 86,42" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    """),

    "slack_agent": ("communication", """
    <!-- Slack-like hash/pound -->
    <line x1="50" y1="34" x2="46" y2="82" stroke="{ic}" stroke-width="5" stroke-linecap="round"/>
    <line x1="70" y1="34" x2="66" y2="82" stroke="{ic}" stroke-width="5" stroke-linecap="round"/>
    <line x1="36" y1="50" x2="82" y2="50" stroke="{ic}" stroke-width="5" stroke-linecap="round"/>
    <line x1="34" y1="68" x2="80" y2="68" stroke="{ic}" stroke-width="5" stroke-linecap="round"/>
    """),

    "whatsapp_agent": ("communication", """
    <!-- Phone/handset -->
    <path d="M60,30 C44,30 34,42 34,56 C34,62 36,68 40,72 L36,84 L50,78 C53,80 56,80 60,80 C76,80 86,68 86,56 C86,42 76,30 60,30 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <path d="M50,50 C50,46 54,44 56,48 L58,52 C58,54 56,55 55,54 C53,56 56,62 60,64 C60,62 62,62 63,64 L66,66 C68,68 66,72 62,70 C54,66 48,58 50,50 Z" fill="{ic}"/>
    """),

    "calendar_agent": ("communication", """
    <!-- Calendar -->
    <rect x="36" y="38" width="48" height="44" rx="4" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="36" y1="52" x2="84" y2="52" stroke="{ic}" stroke-width="3"/>
    <line x1="48" y1="32" x2="48" y2="42" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="72" y1="32" x2="72" y2="42" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="52" cy="64" r="3" fill="{ic}"/>
    <circle cx="64" cy="64" r="3" fill="{ic}"/>
    <circle cx="52" cy="74" r="3" fill="{ic}"/>
    """),

    "communication_memory": ("communication", """
    <!-- Chat bubble with dots -->
    <path d="M38,38 L82,38 C84,38 86,40 86,42 L86,66 C86,68 84,70 82,70 L52,70 L42,82 L42,70 L38,70 C36,70 34,68 34,66 L34,42 C34,40 36,38 38,38 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <circle cx="50" cy="54" r="3" fill="{ic}"/>
    <circle cx="62" cy="54" r="3" fill="{ic}"/>
    <circle cx="74" cy="54" r="3" fill="{ic}"/>
    """),

    # ═══ Content ═══
    "content_manager": ("content", """
    <!-- Newspaper/article -->
    <rect x="36" y="32" width="48" height="56" rx="4" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="44" y1="42" x2="76" y2="42" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <line x1="44" y1="52" x2="62" y2="52" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="44" y1="60" x2="70" y2="60" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="44" y1="68" x2="58" y2="68" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <rect x="66" y="52" width="12" height="16" rx="1" fill="{ic}" opacity="0.4"/>
    """),

    "article_researcher": ("content", """
    <!-- Magnifying glass -->
    <circle cx="54" cy="52" r="18" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="67" y1="65" x2="82" y2="80" stroke="{ic}" stroke-width="4.5" stroke-linecap="round"/>
    """),

    "article_writer": ("content", """
    <!-- Pen/quill -->
    <path d="M72,30 L82,40 L50,72 L38,76 L42,64 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <line x1="64" y1="38" x2="74" y2="48" stroke="{ic}" stroke-width="2.5"/>
    <line x1="38" y1="82" x2="56" y2="82" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    """),

    "social_media_manager": ("content", """
    <!-- Megaphone -->
    <path d="M40,48 L40,68 L50,68 L50,48 Z" fill="{ic}" opacity="0.5"/>
    <path d="M50,42 L80,30 L80,80 L50,68 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <line x1="40" y1="48" x2="50" y2="48" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="40" y1="68" x2="50" y2="68" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="44" y1="68" x2="48" y2="82" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    """),

    "content_memory": ("content", """
    <!-- Book stack -->
    <rect x="38" y="32" width="44" height="10" rx="2" fill="none" stroke="{ic}" stroke-width="3" transform="rotate(-3 60 37)"/>
    <rect x="36" y="46" width="48" height="10" rx="2" fill="none" stroke="{ic}" stroke-width="3"/>
    <rect x="38" y="60" width="44" height="10" rx="2" fill="none" stroke="{ic}" stroke-width="3" transform="rotate(2 60 65)"/>
    <rect x="40" y="74" width="40" height="10" rx="2" fill="none" stroke="{ic}" stroke-width="3" transform="rotate(-1 60 79)"/>
    """),

    # ═══ Development ═══
    "development_manager": ("development", """
    <!-- Wrench + gear -->
    <circle cx="56" cy="48" r="14" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <circle cx="56" cy="48" r="5" fill="{ic}"/>
    <path d="M68,60 L82,74 C84,76 84,80 82,82 C80,84 76,84 74,82 L60,68" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
    """),

    "backend_developer": ("development", """
    <!-- Server rack -->
    <rect x="38" y="30" width="44" height="16" rx="3" fill="none" stroke="{ic}" stroke-width="3"/>
    <rect x="38" y="50" width="44" height="16" rx="3" fill="none" stroke="{ic}" stroke-width="3"/>
    <rect x="38" y="70" width="44" height="16" rx="3" fill="none" stroke="{ic}" stroke-width="3"/>
    <circle cx="72" cy="38" r="3" fill="{ic}"/>
    <circle cx="72" cy="58" r="3" fill="{ic}"/>
    <circle cx="72" cy="78" r="3" fill="{ic}"/>
    <line x1="46" y1="38" x2="62" y2="38" stroke="{ic}" stroke-width="2" stroke-linecap="round"/>
    <line x1="46" y1="58" x2="58" y2="58" stroke="{ic}" stroke-width="2" stroke-linecap="round"/>
    <line x1="46" y1="78" x2="56" y2="78" stroke="{ic}" stroke-width="2" stroke-linecap="round"/>
    """),

    "frontend_developer": ("development", """
    <!-- Browser window -->
    <rect x="34" y="32" width="52" height="52" rx="4" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="34" y1="46" x2="86" y2="46" stroke="{ic}" stroke-width="2.5"/>
    <circle cx="42" cy="39" r="2.5" fill="{ic}"/>
    <circle cx="50" cy="39" r="2.5" fill="{ic}"/>
    <circle cx="58" cy="39" r="2.5" fill="{ic}"/>
    <polyline points="48,58 42,64 48,70" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <polyline points="68,58 74,64 68,70" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="56" y1="56" x2="62" y2="72" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    """),

    "database_manager": ("development", """
    <!-- Database cylinder -->
    <ellipse cx="60" cy="38" rx="22" ry="10" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="38" y1="38" x2="38" y2="78" stroke="{ic}" stroke-width="3.5"/>
    <line x1="82" y1="38" x2="82" y2="78" stroke="{ic}" stroke-width="3.5"/>
    <ellipse cx="60" cy="78" rx="22" ry="10" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <ellipse cx="60" cy="58" rx="22" ry="10" fill="none" stroke="{ic}" stroke-width="2" opacity="0.5"/>
    """),

    "devops": ("development", """
    <!-- Infinity/loop (CI/CD) -->
    <path d="M36,56 C36,44 46,36 56,44 L64,50 C74,58 84,50 84,44 C84,38 78,34 72,34 C66,34 62,38 64,44" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <path d="M84,56 C84,68 74,76 64,68 L56,62 C46,54 36,62 36,68 C36,74 42,78 48,78 C54,78 58,74 56,68" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    """),

    "development_memory": ("development", """
    <!-- Git branch -->
    <circle cx="46" cy="40" r="5" fill="none" stroke="{ic}" stroke-width="3"/>
    <circle cx="74" cy="50" r="5" fill="none" stroke="{ic}" stroke-width="3"/>
    <circle cx="46" cy="76" r="5" fill="none" stroke="{ic}" stroke-width="3"/>
    <line x1="46" y1="45" x2="46" y2="71" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <path d="M46,55 C46,50 60,50 74,50" fill="none" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    """),

    # ═══ Finance ═══
    "finance_manager": ("finance", """
    <!-- Dollar sign in circle -->
    <circle cx="60" cy="58" r="24" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <path d="M54,68 C54,72 58,74 62,74 C66,74 70,72 70,68 C70,62 54,62 54,54 C54,50 58,46 62,46 C66,46 70,48 70,52" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="62" y1="42" x2="62" y2="78" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    """),

    "accounting_agent": ("finance", """
    <!-- Calculator -->
    <rect x="38" y="30" width="44" height="56" rx="4" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <rect x="44" y="36" width="32" height="14" rx="2" fill="{ic}" opacity="0.4"/>
    <circle cx="50" cy="60" r="3" fill="{ic}"/>
    <circle cx="62" cy="60" r="3" fill="{ic}"/>
    <circle cx="74" cy="60" r="3" fill="{ic}"/>
    <circle cx="50" cy="72" r="3" fill="{ic}"/>
    <circle cx="62" cy="72" r="3" fill="{ic}"/>
    <rect x="70" y="68" width="8" height="8" rx="1" fill="{ic}"/>
    """),

    "crypto_agent": ("finance", """
    <!-- Bitcoin-style B -->
    <circle cx="60" cy="58" r="24" fill="none" stroke="{ic}" stroke-width="3"/>
    <path d="M52,44 L52,72 L62,72 C70,72 74,68 74,64 C74,60 70,58 66,58 C72,58 74,54 74,50 C74,46 70,44 62,44 L52,44 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <line x1="52" y1="58" x2="68" y2="58" stroke="{ic}" stroke-width="2.5"/>
    <line x1="56" y1="40" x2="56" y2="44" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="64" y1="40" x2="64" y2="44" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="56" y1="72" x2="56" y2="76" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="64" y1="72" x2="64" y2="76" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    """),

    "finance_memory": ("finance", """
    <!-- Trending chart -->
    <polyline points="36,76 50,60 60,66 76,40 84,44" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
    <polyline points="70,40 82,40 82,50" fill="none" stroke="{ic}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="36" y1="82" x2="84" y2="82" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="36" y1="34" x2="36" y2="82" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    """),

    # ═══ Grant Writing ═══
    "grant_manager": ("grant_writing", """
    <!-- Trophy -->
    <path d="M46,34 L46,54 C46,66 52,72 60,72 C68,72 74,66 74,54 L74,34 Z" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <path d="M46,42 L38,42 C34,42 32,48 36,52 L46,52" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M74,42 L82,42 C86,42 88,48 84,52 L74,52" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="60" y1="72" x2="60" y2="80" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <line x1="48" y1="82" x2="72" y2="82" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    """),

    "grant_writer": ("grant_writing", """
    <!-- Document with pen -->
    <rect x="40" y="30" width="36" height="50" rx="3" fill="none" stroke="{ic}" stroke-width="3"/>
    <line x1="48" y1="42" x2="68" y2="42" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="48" y1="52" x2="64" y2="52" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="48" y1="62" x2="60" y2="62" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M72,60 L82,50 L86,54 L76,64 L70,66 Z" fill="{ic}" stroke="{ic}" stroke-width="2" stroke-linejoin="round"/>
    """),

    "budget_manager": ("grant_writing", """
    <!-- Pie chart -->
    <circle cx="58" cy="58" r="22" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="58" y1="36" x2="58" y2="58" stroke="{ic}" stroke-width="3"/>
    <line x1="58" y1="58" x2="78" y2="50" stroke="{ic}" stroke-width="3"/>
    <path d="M58,36 A22,22 0 0,1 78,50" fill="{ic}" opacity="0.3"/>
    <line x1="58" y1="58" x2="42" y2="72" stroke="{ic}" stroke-width="3"/>
    """),

    "project_manager": ("grant_writing", """
    <!-- Gantt chart / timeline -->
    <line x1="38" y1="36" x2="38" y2="82" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <rect x="44" y="36" width="30" height="8" rx="2" fill="{ic}"/>
    <rect x="44" y="50" width="22" height="8" rx="2" fill="{ic}" opacity="0.7"/>
    <rect x="44" y="64" width="36" height="8" rx="2" fill="{ic}" opacity="0.5"/>
    <rect x="44" y="78" width="18" height="8" rx="2" fill="{ic}" opacity="0.3"/>
    <line x1="62" y1="32" x2="62" y2="90" stroke="{ic}" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/>
    """),

    "grant_memory": ("grant_writing", """
    <!-- Filing cabinet -->
    <rect x="38" y="30" width="44" height="20" rx="3" fill="none" stroke="{ic}" stroke-width="3"/>
    <rect x="38" y="54" width="44" height="20" rx="3" fill="none" stroke="{ic}" stroke-width="3"/>
    <rect x="38" y="78" width="44" height="8" rx="3" fill="none" stroke="{ic}" stroke-width="3" opacity="0.5"/>
    <line x1="56" y1="38" x2="64" y2="38" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <line x1="56" y1="62" x2="64" y2="62" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    """),

    # ═══ Research ═══
    "research_manager": ("research", """
    <!-- Flask/beaker -->
    <path d="M52,30 L52,52 L36,78 C34,82 36,86 40,86 L80,86 C84,86 86,82 84,78 L68,52 L68,30" fill="none" stroke="{ic}" stroke-width="3.5" stroke-linejoin="round"/>
    <line x1="48" y1="30" x2="72" y2="30" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <ellipse cx="60" cy="72" rx="12" ry="6" fill="{ic}" opacity="0.35"/>
    <circle cx="54" cy="68" r="3" fill="{ic}" opacity="0.5"/>
    <circle cx="66" cy="74" r="2" fill="{ic}" opacity="0.5"/>
    """),

    "media_platform_researcher": ("research", """
    <!-- TV/monitor -->
    <rect x="34" y="32" width="52" height="38" rx="4" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <line x1="60" y1="70" x2="60" y2="80" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <line x1="46" y1="82" x2="74" y2="82" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <polygon points="54,44 54,60 68,52" fill="{ic}"/>
    """),

    "media_program_researcher": ("research", """
    <!-- Graduation cap -->
    <polygon points="60,34 30,50 60,66 90,50" fill="none" stroke="{ic}" stroke-width="3" stroke-linejoin="round"/>
    <path d="M42,56 L42,72 C42,78 50,82 60,82 C70,82 78,78 78,72 L78,56" fill="none" stroke="{ic}" stroke-width="3" stroke-linejoin="round"/>
    <line x1="86" y1="52" x2="86" y2="76" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    """),

    "street_bot_researcher": ("research", """
    <!-- Robot face -->
    <rect x="38" y="38" width="44" height="36" rx="6" fill="none" stroke="{ic}" stroke-width="3.5"/>
    <circle cx="50" cy="54" r="4" fill="{ic}"/>
    <circle cx="70" cy="54" r="4" fill="{ic}"/>
    <line x1="50" y1="66" x2="70" y2="66" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <line x1="60" y1="30" x2="60" y2="38" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <circle cx="60" cy="28" r="3" fill="{ic}"/>
    <line x1="34" y1="50" x2="38" y2="50" stroke="{ic}" stroke-width="4" stroke-linecap="round"/>
    <line x1="82" y1="50" x2="86" y2="50" stroke="{ic}" stroke-width="4" stroke-linecap="round"/>
    """),

    "research_memory": ("research", """
    <!-- Microscope -->
    <circle cx="56" cy="40" r="10" fill="none" stroke="{ic}" stroke-width="3"/>
    <line x1="56" y1="50" x2="56" y2="72" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="56" y1="62" x2="74" y2="50" stroke="{ic}" stroke-width="3" stroke-linecap="round"/>
    <line x1="42" y1="78" x2="72" y2="78" stroke="{ic}" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="46" y1="72" x2="66" y2="72" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    """),

    # ═══ Scraping ═══
    "scraping_manager": ("scraping", """
    <!-- Spider -->
    <circle cx="60" cy="52" r="10" fill="{ic}" opacity="0.7"/>
    <circle cx="60" cy="42" r="6" fill="{ic}" opacity="0.7"/>
    <!-- legs -->
    <path d="M50,48 L34,36" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M70,48 L86,36" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M48,54 L32,54" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M72,54 L88,54" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M50,58 L36,72" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M70,58 L84,72" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M54,62 L44,80" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M66,62 L76,80" fill="none" stroke="{ic}" stroke-width="2.5" stroke-linecap="round"/>
    """),

    "scraping_agent": ("scraping", """
    <!-- Globe with cursor -->
    <circle cx="56" cy="56" r="22" fill="none" stroke="{ic}" stroke-width="3"/>
    <ellipse cx="56" cy="56" rx="10" ry="22" fill="none" stroke="{ic}" stroke-width="2"/>
    <line x1="34" y1="56" x2="78" y2="56" stroke="{ic}" stroke-width="2"/>
    <line x1="56" y1="34" x2="56" y2="78" stroke="{ic}" stroke-width="2"/>
    <!-- cursor arrow -->
    <path d="M72,62 L72,82 L78,76 L84,84 L88,80 L82,72 L88,68 Z" fill="{ic}" stroke="{ic}" stroke-width="1.5"/>
    """),

    "scraper_memory": ("scraping", """
    <!-- Package/box -->
    <path d="M60,30 L86,44 L86,74 L60,88 L34,74 L34,44 Z" fill="none" stroke="{ic}" stroke-width="3" stroke-linejoin="round"/>
    <line x1="60" y1="58" x2="60" y2="88" stroke="{ic}" stroke-width="2.5"/>
    <line x1="34" y1="44" x2="60" y2="58" stroke="{ic}" stroke-width="2.5"/>
    <line x1="86" y1="44" x2="60" y2="58" stroke="{ic}" stroke-width="2.5"/>
    """),
}

# Unused old file
SKIP_FILES = {"communication_manager_icon"}


def generate_svg(name: str, team: str, icon_snippet: str) -> str:
    g_start, g_end, ic = TEAM_COLORS[team]

    icon_filled = icon_snippet.replace("{ic}", ic)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">
  <defs>
    <linearGradient id="g_{name}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{g_start}"/>
      <stop offset="100%" stop-color="{g_end}"/>
    </linearGradient>
  </defs>
  <circle cx="60" cy="60" r="56" fill="url(#g_{name})"/>
  <g>{icon_filled}
  </g>
</svg>
"""


def main():
    count = 0
    for name, (team, icon) in AGENTS.items():
        svg = generate_svg(name, team, icon)
        out = OUTPUT_DIR / f"{name}.svg"
        out.write_text(svg)
        count += 1
        print(f"  {name}.svg ({team})")

    print(f"\nGenerated {count} SVG avatars")


if __name__ == "__main__":
    main()
