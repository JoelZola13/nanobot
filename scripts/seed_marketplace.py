"""Seed LibreChat marketplace with all nanobot agents.

Reads agent specs from YAML files, loads system prompts from .md files,
and inserts agent documents + public ACL entries into LibreChat's MongoDB.

Usage:
    python scripts/seed_marketplace.py [--dry-run]
"""

import sys
import string
import random
from pathlib import Path
from datetime import datetime, timezone

import yaml
from pymongo import MongoClient
from bson import ObjectId


# ── Config ──────────────────────────────────────────────────────────────────

MONGO_HOST = "localhost"
MONGO_PORT = 27018
MONGO_DB = "LibreChat"
AUTHOR_OID = ObjectId("6996680a024e2772291b17cd")  # Joel (nanobot-mongodb)

TEAMS_DIR = Path(__file__).resolve().parent.parent / "nanobot" / "agents" / "teams"

# Provider/model that LibreChat will show (these are display values)
PROVIDER = "Nanobot"
MODEL = "openai-codex/gpt-5.4"

# MDI icon CDN base URL (matches librechat.yaml iconURL fields)
MDI_BASE = "https://cdn.jsdelivr.net/npm/@mdi/svg@7.4.47/svg"

# Map agent names → MDI icon filenames (must match librechat.yaml iconURL)
AGENT_ICON_MAP = {
    # Executive
    "ceo": "account-tie.svg",
    "executive_memory": "brain.svg",
    "security_compliance": "shield-check.svg",
    # Communication
    "communication_manager": "http://localhost:18790/avatars/communication_manager_icon.svg",
    "email_agent": "email.svg",
    "slack_agent": "slack.svg",
    "whatsapp_agent": "whatsapp.svg",
    "calendar_agent": "calendar-clock.svg",
    "communication_memory": "database-sync.svg",
    # Content
    "content_manager": "newspaper-variant-multiple.svg",
    "article_researcher": "book-search.svg",
    "article_writer": "typewriter.svg",
    "social_media_manager": "share-variant.svg",
    "content_memory": "database-sync.svg",
    # Development
    "development_manager": "code-braces-box.svg",
    "backend_developer": "server.svg",
    "frontend_developer": "monitor-dashboard.svg",
    "database_manager": "database.svg",
    "devops": "cloud-cog.svg",
    "development_memory": "database-sync.svg",
    # Finance
    "finance_manager": "finance.svg",
    "accounting_agent": "calculator-variant.svg",
    "crypto_agent": "bitcoin.svg",
    "finance_memory": "database-sync.svg",
    # Grant Writing
    "grant_manager": "file-document-edit.svg",
    "grant_writer": "text-box-edit.svg",
    "budget_manager": "cash-multiple.svg",
    "project_manager": "clipboard-check.svg",
    "grant_memory": "database-sync.svg",
    # Research
    "research_manager": "magnify-scan.svg",
    "media_platform_researcher": "cellphone-link.svg",
    "media_program_researcher": "television-classic.svg",
    "street_bot_researcher": "map-marker-radius.svg",
    "research_memory": "database-sync.svg",
    # Scraping
    "scraping_manager": "spider-web.svg",
    "scraping_agent": "web.svg",
    "scraper_memory": "database-sync.svg",
}

# Map nanobot team names → LibreChat category values
TEAM_CATEGORY_MAP = {
    "executive": "executive",
    "communication": "communication",
    "content": "content",
    "development": "development",
    "finance": "finance",
    "grant_writing": "grant_writing",
    "research": "research",
    "scraping": "scraping",
}

# Human-readable labels for custom categories
CATEGORY_LABELS = {
    "executive": "Executive & Leadership",
    "communication": "Communication",
    "content": "Content & Media",
    "development": "Development & Engineering",
    "finance": "Finance & Accounting",
    "grant_writing": "Grant Writing & Projects",
    "research": "Research & Intelligence",
    "scraping": "Scraping & Data Extraction",
}

# Tool display names for the marketplace
TOOL_DISPLAY = {
    "web_search": "Web Search",
    "web_fetch": "Web Fetch",
    "news_search": "News Search",
    "image_search": "Image Search",
    "academic_search": "Academic Search",
    "file_read": "File Read",
    "file_write": "File Write",
    "file_edit": "File Edit",
    "edit_file": "File Edit",
    "list_dir": "List Directory",
    "shell": "Shell",
    "exec": "Shell",
    "email_send": "Email Send",
    "email_read": "Email Read",
    "image_gen": "Image Generation",
    "tts": "Text-to-Speech",
    "postiz": "Social Publishing",
    "article_image": "Article Images",
    "web_scrape": "Web Scrape",
    "html_parser": "HTML Parser",
    "data_extractor": "Data Extractor",
    "calculator": "Calculator",
    "crypto_api": "Crypto API",
    "invoice_generator": "Invoice Generator",
    "spreadsheet_read": "Spreadsheet Read",
    "spreadsheet_write": "Spreadsheet Write",
    "document_editor": "Document Editor",
    "calendar_read": "Calendar Read",
    "grants_database": "Grants Database",
}


# ── Rich Agent Profiles ────────────────────────────────────────────────────
# Comprehensive profile data for each of the 37 agents.
# Used to build detailed marketplace descriptions and instructions.

AGENT_PROFILES = {
    # ── Executive Team ──────────────────────────────────────────────────────
    "ceo": {
        "tagline": "Chief orchestrator of all nanobot operations",
        "expertise": ["Strategic Planning", "Cross-Team Coordination", "Task Delegation", "Priority Management", "Organizational Oversight", "Decision Making"],
        "capabilities": "Serves as the central intelligence hub for Street Voices' entire nanobot ecosystem. Receives all incoming requests, assesses complexity and urgency, and routes work to the appropriate team lead. Maintains a bird's-eye view of active projects across all eight teams. Resolves conflicts between competing priorities and escalates only when human judgment is required. Ensures every task gets a clear owner and deadline.",
        "use_cases": [
            "Coordinate a multi-team project like a grant application launch",
            "Triage a batch of incoming requests by priority",
            "Get a status update across all active projects",
            "Decide which team should handle an ambiguous request",
            "Escalate a blocker that needs Joel's attention",
        ],
        "integrations": ["All team leads via handoffs", "Slack DM delivery", "CronService scheduling"],
    },
    "security_compliance": {
        "tagline": "Guardian of data privacy and operational safety",
        "expertise": ["Data Privacy", "Access Control", "Compliance Auditing", "Risk Assessment", "Incident Response", "Policy Enforcement"],
        "capabilities": "Monitors all nanobot operations for security risks and compliance violations. Reviews outbound communications before they leave the system. Enforces consent rules: no emails, messages, or posts are sent without explicit human approval. Audits tool access patterns and flags anomalies. Maintains the organization's data handling policies aligned with Canadian privacy regulations (PIPEDA) and arts council requirements.",
        "use_cases": [
            "Review an outbound email for compliance before sending",
            "Audit which agents accessed sensitive data this week",
            "Check if a new integration meets privacy requirements",
            "Flag a potential data leak or unauthorized action",
            "Generate a compliance report for a grant funder",
        ],
        "integrations": ["All agent logs", "Email audit trail", "Airtable access logs"],
    },
    "executive_memory": {
        "tagline": "Institutional knowledge keeper for leadership decisions",
        "expertise": ["Decision Archival", "Pattern Recognition", "Historical Context", "Cross-Reference Indexing", "Strategic Memory", "Organizational Learning"],
        "capabilities": "Maintains the long-term memory of all executive-level decisions, preferences, and organizational patterns. Indexes Joel's standing instructions, recurring workflows, and lessons learned. Provides historical context when similar situations arise — preventing repeated mistakes and reinforcing successful patterns. Cross-references decisions across teams to maintain consistency.",
        "use_cases": [
            "Recall Joel's preference for a specific workflow",
            "Find the last time a similar decision was made",
            "Surface lessons learned from a past project",
            "Track recurring patterns in organizational requests",
            "Provide context for a decision that was made months ago",
        ],
        "integrations": ["Memory store", "Event history log", "All team memory agents"],
    },

    # ── Communication Team ──────────────────────────────────────────────────
    "communication_manager": {
        "tagline": "Orchestrates all messaging channels with precision",
        "expertise": ["Multi-Channel Coordination", "Message Routing", "Response Prioritization", "Tone Calibration", "Communication Strategy", "Stakeholder Management"],
        "capabilities": "Leads the communication team, routing messages across email, Slack, WhatsApp, and calendar channels. Determines the best channel and timing for each outbound message. Ensures consistent Street Voices voice across all platforms — warm, community-first, and professional. Manages communication queues, prioritizes urgent messages, and coordinates multi-touch outreach sequences.",
        "use_cases": [
            "Plan a multi-channel outreach campaign to partners",
            "Decide the best channel to reach a specific contact",
            "Coordinate a series of follow-up messages across platforms",
            "Review and approve a batch of draft communications",
            "Set up an automated check-in schedule with a funder",
        ],
        "integrations": ["Email Agent", "Slack Agent", "WhatsApp Agent", "Calendar Agent", "Airtable Contacts"],
    },
    "email_agent": {
        "tagline": "Your inbox, intelligently managed",
        "expertise": ["Gmail Integration", "Draft Composition", "Email Triage", "Thread Management", "Contact Recognition", "Follow-up Tracking"],
        "capabilities": "Manages all email communications through Gmail for joel@streetvoices.ca. Drafts messages in Street Voices' warm, community-first voice. Reads and triages inbox by priority — flagging messages from funders, city officials, and partners. Tracks conversation threads and follow-up deadlines. Never sends without explicit approval from Joel.",
        "use_cases": [
            "Draft a follow-up email to a grant officer",
            "Check inbox for anything from the city council",
            "Summarize unread emails from this week",
            "Compose a partnership outreach email",
            "Track which emails need a response by Friday",
        ],
        "integrations": ["Gmail IMAP/SMTP", "Airtable Contacts", "Calendar sync"],
    },
    "slack_agent": {
        "tagline": "Real-time team communication hub",
        "expertise": ["Slack Messaging", "Channel Management", "Thread Coordination", "Notification Routing", "Bot Interaction", "Status Updates"],
        "capabilities": "Handles all Slack communications for Street Voices. Posts updates to relevant channels, manages thread conversations, and routes notifications. Delivers cron job results, WhatsApp message forwards, and system alerts to Joel's DM. Maintains awareness of channel context and conversation history for informed responses.",
        "use_cases": [
            "Post a project update to a team channel",
            "Send Joel a summary via Slack DM",
            "Check recent messages in a specific channel",
            "Forward an important notification to the right channel",
            "Coordinate a quick decision via Slack thread",
        ],
        "integrations": ["Slack Bot API", "Socket Mode", "Channel webhooks"],
    },
    "whatsapp_agent": {
        "tagline": "Personal messaging with community partners",
        "expertise": ["WhatsApp Messaging", "Voice Transcription", "Contact Management", "Message Formatting", "Media Handling", "Delivery Tracking"],
        "capabilities": "Manages WhatsApp communications for personal and community outreach. Sends messages only with Joel's explicit approval — never auto-replies. Transcribes incoming voice messages using Groq. Forwards important incoming messages to Slack for visibility. Handles media attachments and maintains conversation context across sessions.",
        "use_cases": [
            "Send a WhatsApp message to a community partner",
            "Transcribe a voice message from a contact",
            "Check recent WhatsApp conversations",
            "Forward an important WhatsApp message to Slack",
            "Draft a WhatsApp message for Joel's review",
        ],
        "integrations": ["Baileys WhatsApp Bridge", "Groq Transcription", "Slack forwarding"],
    },
    "calendar_agent": {
        "tagline": "Schedule master for meetings and deadlines",
        "expertise": ["Google Calendar", "Event Scheduling", "Conflict Detection", "Reminder Management", "Availability Checking", "Timezone Handling"],
        "capabilities": "Manages Joel's Google Calendar for Street Voices operations. Creates, updates, and monitors events. Detects scheduling conflicts before they happen. Provides daily briefings of upcoming meetings and deadlines. Handles timezone conversions for contacts across Canada and internationally. Coordinates with email and Slack agents for meeting-related communications.",
        "use_cases": [
            "Schedule a meeting with a grant officer next week",
            "Check availability for a Tuesday afternoon call",
            "Get today's calendar briefing",
            "Set a deadline reminder for a grant submission",
            "Find a free slot for a team check-in this week",
        ],
        "integrations": ["Google Calendar API", "Gmail", "Slack notifications"],
    },
    "communication_memory": {
        "tagline": "Communication patterns and contact intelligence",
        "expertise": ["Contact History", "Communication Patterns", "Channel Preferences", "Response Tracking", "Relationship Mapping", "Tone Calibration Memory"],
        "capabilities": "Maintains the long-term memory of all communication activities across email, Slack, WhatsApp, and calendar. Tracks contact preferences — who prefers email vs phone, optimal send times, and conversation history summaries. Remembers follow-up commitments and flags overdue responses. Builds a relationship map of key contacts and their communication patterns.",
        "use_cases": [
            "Recall when we last contacted a specific funder",
            "Find the preferred communication channel for a contact",
            "Track overdue follow-ups across all channels",
            "Surface patterns in response times from city officials",
            "Remember the tone and style used with a specific partner",
        ],
        "integrations": ["Memory store", "Event history log", "All communication agents"],
    },

    # ── Content Team ────────────────────────────────────────────────────────
    "content_manager": {
        "tagline": "Editorial director for all Street Voices content",
        "expertise": ["Content Strategy", "Editorial Planning", "Quality Control", "Brand Voice", "Multi-Format Publishing", "Content Calendar"],
        "capabilities": "Leads the content team, overseeing all content creation from research through publication. Maintains the editorial calendar and ensures consistent Street Voices voice across articles, social posts, and newsletters. Assigns research and writing tasks, reviews drafts for quality and brand alignment, and coordinates publication timing across platforms.",
        "use_cases": [
            "Plan this month's content calendar",
            "Review a draft article before publication",
            "Assign a research topic to the article researcher",
            "Coordinate a social media campaign around an event",
            "Ensure brand voice consistency across recent posts",
        ],
        "integrations": ["Article Researcher", "Article Writer", "Social Media Manager", "Airtable Content Calendar"],
    },
    "article_researcher": {
        "tagline": "Deep-dive researcher for compelling stories",
        "expertise": ["Web Research", "Source Verification", "Data Gathering", "Trend Analysis", "Academic Search", "Interview Preparation"],
        "capabilities": "Conducts thorough research for Street Voices articles and content pieces. Searches web, news, and academic sources to gather facts, statistics, and quotes. Verifies source credibility and cross-references claims. Prepares research briefs with key findings, relevant data points, and suggested angles for the article writer. Specializes in community media, arts funding, and Toronto cultural landscape topics.",
        "use_cases": [
            "Research community radio trends in Canada for an article",
            "Find recent statistics on arts funding in Ontario",
            "Gather background on a potential interview subject",
            "Compile a research brief on digital media literacy programs",
            "Verify claims in a draft article with primary sources",
        ],
        "integrations": ["Brave Web Search", "News Search", "Academic Search", "Web Fetch"],
    },
    "article_writer": {
        "tagline": "Storyteller crafting the Street Voices narrative",
        "expertise": ["Article Writing", "Blog Posts", "Newsletter Copy", "SEO Optimization", "Headline Crafting", "Story Structure"],
        "capabilities": "Writes compelling articles, blog posts, and newsletter content in Street Voices' distinctive voice. Transforms research briefs into engaging narratives that resonate with community media audiences. Optimizes content for readability and SEO without sacrificing authenticity. Crafts attention-grabbing headlines and structures stories with clear narrative arcs. Produces drafts for Joel's review before publication.",
        "use_cases": [
            "Write a blog post about a community radio workshop",
            "Draft the monthly newsletter feature article",
            "Create a compelling headline for an upcoming piece",
            "Rewrite a technical report as an accessible story",
            "Draft talking points for a media appearance",
        ],
        "integrations": ["File Write", "Content Calendar", "Image Generation"],
    },
    "social_media_manager": {
        "tagline": "Amplifying Street Voices across social platforms",
        "expertise": ["Social Media Strategy", "Post Scheduling", "Hashtag Research", "Community Engagement", "Analytics Tracking", "Platform Optimization"],
        "capabilities": "Manages Street Voices' social media presence across platforms. Creates engaging posts optimized for each platform's format and audience. Schedules content to maximize reach and engagement. Researches trending hashtags and topics relevant to community media. Drafts social threads, image captions, and video descriptions. Tracks engagement metrics and adjusts strategy based on performance.",
        "use_cases": [
            "Draft a Twitter thread about an upcoming event",
            "Schedule social posts for this week's content",
            "Research trending hashtags for community media",
            "Create Instagram captions for workshop photos",
            "Draft a LinkedIn post about a partnership announcement",
        ],
        "integrations": ["Postiz Social Publishing", "Image Generation", "Content Calendar"],
    },
    "content_memory": {
        "tagline": "Editorial history and content intelligence",
        "expertise": ["Content Archive", "Performance Tracking", "Topic History", "Voice Consistency", "Publishing Patterns", "Audience Insights"],
        "capabilities": "Maintains the long-term memory of all content operations. Tracks which topics have been covered, their performance metrics, and audience reception. Remembers editorial decisions, style preferences, and lessons learned from past publications. Provides historical context for content planning — preventing topic repetition and surfacing successful patterns.",
        "use_cases": [
            "Check if we've covered a topic before",
            "Recall which article formats perform best",
            "Find the style guide decision for a specific term",
            "Surface content gaps in our coverage",
            "Track the evolution of our editorial voice",
        ],
        "integrations": ["Memory store", "Event history log", "All content agents"],
    },

    # ── Development Team ────────────────────────────────────────────────────
    "development_manager": {
        "tagline": "Technical lead for the nanobot platform",
        "expertise": ["Architecture Planning", "Sprint Management", "Code Review", "Technical Debt", "Integration Design", "Team Coordination"],
        "capabilities": "Leads the development team responsible for maintaining and extending the nanobot platform. Plans technical architecture, assigns development tasks, and reviews code quality. Manages the integration pipeline between nanobot and external services (LibreChat, MCP servers, APIs). Tracks technical debt and prioritizes fixes alongside new features. Coordinates deployments and ensures system stability.",
        "use_cases": [
            "Plan the architecture for a new integration",
            "Assign a bug fix to the right developer",
            "Review a pull request for code quality",
            "Prioritize the technical debt backlog",
            "Coordinate a deployment across services",
        ],
        "integrations": ["Backend Developer", "Frontend Developer", "Database Manager", "DevOps", "GitHub"],
    },
    "backend_developer": {
        "tagline": "Python powerhouse building the agent engine",
        "expertise": ["Python Development", "API Design", "Starlette/FastAPI", "Agent Framework", "MCP Integration", "SSE Streaming"],
        "capabilities": "Develops and maintains the nanobot backend — the Python-based agent framework, API server, and tool integrations. Writes clean, efficient Python code for new agent capabilities, API endpoints, and service integrations. Handles the Starlette API bridge between LibreChat and the nanobot AgentLoop. Implements SSE streaming, tool registration, and MCP server connections.",
        "use_cases": [
            "Add a new tool to the agent framework",
            "Fix a bug in the API streaming endpoint",
            "Implement a new MCP server integration",
            "Optimize the agent loop for better performance",
            "Write a new API endpoint for a feature",
        ],
        "integrations": ["Python 3.12", "Starlette", "uvicorn", "MCP SDK", "OpenAI API"],
    },
    "frontend_developer": {
        "tagline": "Crafting the user experience layer",
        "expertise": ["React/TypeScript", "UI Components", "LibreChat Customization", "Responsive Design", "Accessibility", "CSS/Tailwind"],
        "capabilities": "Develops and customizes the LibreChat frontend for Street Voices' needs. Builds React components, modifies the agent marketplace UI, and ensures responsive design across devices. Implements accessibility best practices and customizes the chat interface. Works with Tailwind CSS for styling and ensures consistent visual identity across the platform.",
        "use_cases": [
            "Customize the agent marketplace layout",
            "Build a new UI component for the dashboard",
            "Fix a responsive design issue on mobile",
            "Improve the accessibility of the chat interface",
            "Style a new feature to match the design system",
        ],
        "integrations": ["React", "TypeScript", "Tailwind CSS", "LibreChat frontend"],
    },
    "database_manager": {
        "tagline": "Data architect for persistent intelligence",
        "expertise": ["MongoDB Administration", "Schema Design", "Query Optimization", "Data Migration", "Backup Strategy", "Airtable Management"],
        "capabilities": "Manages all database operations across MongoDB (LibreChat) and Airtable (business data). Designs schemas for efficient data storage and retrieval. Optimizes queries for performance. Handles data migrations when schemas evolve. Maintains backup strategies and data integrity. Manages the Airtable bases that power Street Voices' business operations.",
        "use_cases": [
            "Optimize a slow MongoDB query",
            "Design a schema for a new data collection",
            "Migrate data between Airtable bases",
            "Set up automated backups for critical data",
            "Query Airtable for a specific business report",
        ],
        "integrations": ["MongoDB", "Airtable API", "Data export tools"],
    },
    "devops": {
        "tagline": "Infrastructure reliability and deployment automation",
        "expertise": ["PM2 Management", "Docker/OrbStack", "Process Monitoring", "Log Analysis", "CI/CD", "Server Administration"],
        "capabilities": "Manages the infrastructure that keeps nanobot running reliably. Oversees PM2 process management, Docker containers via OrbStack, and server health monitoring. Handles deployments, restarts, and log analysis. Ensures all services (nanobot API, WhatsApp bridge, LibreChat, MongoDB) stay healthy and recover from failures. Manages port allocation and prevents conflicts.",
        "use_cases": [
            "Restart the nanobot API after a code change",
            "Diagnose why a service crashed from PM2 logs",
            "Check the health of all running services",
            "Deploy a new version of a component",
            "Resolve a port conflict between services",
        ],
        "integrations": ["PM2", "Docker/OrbStack", "systemd", "Log files"],
    },
    "development_memory": {
        "tagline": "Technical knowledge base and code intelligence",
        "expertise": ["Code History", "Bug Patterns", "Architecture Decisions", "Dependency Tracking", "Deployment History", "Technical Debt Registry"],
        "capabilities": "Maintains the long-term memory of all development activities. Tracks architecture decisions and their rationale, bug patterns and their fixes, and deployment history. Remembers which approaches worked and which didn't — preventing repeated mistakes. Indexes critical lessons learned (like the auto-reply incident, zombie process fixes, and MCP stale session handling).",
        "use_cases": [
            "Recall why a specific architecture decision was made",
            "Find the fix for a recurring bug pattern",
            "Track the history of a specific code file",
            "Surface technical debt that needs attention",
            "Remember deployment procedures for each service",
        ],
        "integrations": ["Memory store", "Event history log", "Git history", "All dev agents"],
    },

    # ── Finance Team ────────────────────────────────────────────────────────
    "finance_manager": {
        "tagline": "Financial strategist for sustainable operations",
        "expertise": ["Budget Planning", "Financial Reporting", "Cash Flow Management", "Grant Budgets", "Expense Tracking", "Financial Strategy"],
        "capabilities": "Leads the finance team, overseeing all financial operations for Street Voices. Plans budgets for grant applications and organizational operations. Tracks income and expenses across funding sources. Generates financial reports for board meetings, funders, and internal review. Ensures financial compliance with grant requirements and Canadian nonprofit regulations. Coordinates with the grant writing team on budget sections.",
        "use_cases": [
            "Prepare a budget for a new grant application",
            "Generate a quarterly financial report",
            "Track spending against a specific grant budget",
            "Forecast cash flow for the next quarter",
            "Review expenses for compliance with funder requirements",
        ],
        "integrations": ["Accounting Agent", "Crypto Agent", "Airtable Finance Base", "Spreadsheets"],
    },
    "accounting_agent": {
        "tagline": "Meticulous record-keeper for every dollar",
        "expertise": ["Bookkeeping", "Invoice Processing", "Expense Categorization", "Tax Preparation", "Receipt Management", "Reconciliation"],
        "capabilities": "Handles day-to-day accounting operations for Street Voices. Processes invoices, categorizes expenses, and maintains accurate financial records. Prepares data for tax filings and grant financial reports. Reconciles bank statements with recorded transactions. Generates invoices for services and tracks payment status. Ensures all financial data is audit-ready.",
        "use_cases": [
            "Categorize this month's expenses by project",
            "Generate an invoice for a consulting service",
            "Reconcile recent bank transactions",
            "Prepare expense data for tax filing",
            "Track outstanding invoices and payment status",
        ],
        "integrations": ["Airtable Finance Base", "Invoice Generator", "Spreadsheets", "Calculator"],
    },
    "crypto_agent": {
        "tagline": "Digital asset monitor and blockchain analyst",
        "expertise": ["Cryptocurrency Tracking", "Portfolio Monitoring", "Market Analysis", "Transaction History", "DeFi Awareness", "Price Alerts"],
        "capabilities": "Monitors cryptocurrency holdings and blockchain activity relevant to Street Voices. Tracks portfolio value, price movements, and market trends. Provides market analysis summaries without giving financial advice. Monitors specific wallets and transactions. Generates reports on digital asset holdings for organizational records.",
        "use_cases": [
            "Check current portfolio value and recent changes",
            "Summarize crypto market trends this week",
            "Track a specific transaction on the blockchain",
            "Generate a digital asset holdings report",
            "Monitor price movements for held tokens",
        ],
        "integrations": ["Crypto API", "Market data feeds", "Portfolio tracker"],
    },
    "finance_memory": {
        "tagline": "Financial history and budget intelligence",
        "expertise": ["Budget History", "Spending Patterns", "Funder Requirements", "Financial Decisions", "Audit Trail", "Cost Benchmarks"],
        "capabilities": "Maintains the long-term memory of all financial operations. Tracks budget decisions, spending patterns, and funder-specific requirements. Remembers which budget line items have been historically approved or questioned by funders. Provides cost benchmarks from past projects for future planning. Indexes financial lessons learned and compliance requirements.",
        "use_cases": [
            "Recall the budget breakdown from our last OAC grant",
            "Find historical spending patterns for a budget category",
            "Remember funder-specific financial reporting requirements",
            "Surface cost benchmarks from similar past projects",
            "Track the history of a specific budget decision",
        ],
        "integrations": ["Memory store", "Event history log", "All finance agents"],
    },

    # ── Grant Writing Team ──────────────────────────────────────────────────
    "grant_manager": {
        "tagline": "Strategic lead for funding acquisition",
        "expertise": ["Grant Strategy", "Deadline Management", "Funder Relations", "Application Coordination", "Success Metrics", "Portfolio Management"],
        "capabilities": "Leads the grant writing team, developing funding strategy and coordinating the full grant application lifecycle. Identifies funding opportunities aligned with Street Voices' mission. Manages application timelines and ensures deadlines are met. Coordinates between grant writer, budget manager, and project manager for comprehensive submissions. Tracks success rates and adjusts strategy based on outcomes.",
        "use_cases": [
            "Identify upcoming grant opportunities for community media",
            "Plan the timeline for an Ontario Arts Council application",
            "Coordinate the team on a complex multi-section grant",
            "Review a complete application before submission",
            "Analyze our grant success rate and improve strategy",
        ],
        "integrations": ["Grant Writer", "Budget Manager", "Project Manager", "Grants Database", "Airtable"],
    },
    "grant_writer": {
        "tagline": "Compelling narratives that win funding",
        "expertise": ["Proposal Writing", "Impact Narratives", "Needs Assessment", "Logic Models", "Letter of Intent", "Reporting"],
        "capabilities": "Crafts compelling grant proposals and reports for Street Voices. Writes persuasive narratives that connect community media impact to funder priorities. Develops needs assessments backed by data and community voice. Creates logic models and theory of change documents. Writes progress reports and final reports that demonstrate impact. Tailors language and framing to each funder's specific evaluation criteria.",
        "use_cases": [
            "Draft the project narrative for a grant application",
            "Write an impact statement for a funder report",
            "Create a needs assessment with supporting data",
            "Develop a logic model for a new program",
            "Draft a letter of intent for a funding opportunity",
        ],
        "integrations": ["File Write", "Document Editor", "Web Research", "Airtable"],
    },
    "budget_manager": {
        "tagline": "Precision budgets that funders trust",
        "expertise": ["Grant Budgets", "Cost Estimation", "Budget Justification", "Financial Projections", "In-Kind Valuation", "Multi-Year Planning"],
        "capabilities": "Creates detailed, defensible budgets for grant applications and project proposals. Estimates costs based on historical data and current market rates. Writes clear budget justifications that explain each line item. Handles multi-year budget projections and cash flow planning. Calculates in-kind contributions and matching fund requirements. Ensures budgets align with funder guidelines and organizational capacity.",
        "use_cases": [
            "Build a detailed budget for a grant application",
            "Write budget justification notes for each line item",
            "Calculate in-kind contributions for a project",
            "Project multi-year costs for a program expansion",
            "Adjust a budget to meet a funder's maximum amount",
        ],
        "integrations": ["Spreadsheets", "Calculator", "Airtable Finance", "Historical budgets"],
    },
    "project_manager": {
        "tagline": "Turning funded projects into delivered impact",
        "expertise": ["Project Planning", "Timeline Management", "Milestone Tracking", "Risk Assessment", "Stakeholder Coordination", "Deliverable Management"],
        "capabilities": "Plans and tracks project execution for funded Street Voices programs. Creates detailed work plans with milestones, timelines, and deliverables. Monitors progress against grant commitments and flags risks early. Coordinates between internal teams and external partners. Prepares project status updates for funders and the board. Ensures funded activities are completed on time and within budget.",
        "use_cases": [
            "Create a work plan for a newly funded project",
            "Track milestones for an active grant",
            "Prepare a project status update for a funder",
            "Identify risks in an upcoming project phase",
            "Coordinate deliverables across multiple team members",
        ],
        "integrations": ["Airtable Projects", "Calendar", "Spreadsheets", "Document Editor"],
    },
    "grant_memory": {
        "tagline": "Funding intelligence and application archive",
        "expertise": ["Application Archive", "Funder Preferences", "Success Patterns", "Deadline Tracking", "Reviewer Feedback", "Reusable Components"],
        "capabilities": "Maintains the long-term memory of all grant-related activities. Archives past applications with outcomes and reviewer feedback. Tracks funder preferences, evaluation criteria changes, and relationship history. Indexes reusable proposal components (boilerplate language, organizational descriptions, impact metrics). Remembers what worked and what didn't across applications.",
        "use_cases": [
            "Find our last application to a specific funder",
            "Recall reviewer feedback from a past submission",
            "Surface reusable boilerplate for organizational description",
            "Track deadline patterns for recurring grant programs",
            "Remember which impact metrics resonated with a funder",
        ],
        "integrations": ["Memory store", "Event history log", "All grant agents"],
    },

    # ── Research Team ───────────────────────────────────────────────────────
    "research_manager": {
        "tagline": "Intelligence director for strategic insights",
        "expertise": ["Research Strategy", "Competitive Analysis", "Trend Forecasting", "Data Synthesis", "Report Generation", "Knowledge Management"],
        "capabilities": "Leads the research team, directing investigations across media platforms, programs, and emerging technologies. Synthesizes findings from multiple researchers into actionable intelligence. Identifies trends, opportunities, and threats in the community media landscape. Produces strategic briefs that inform organizational decision-making. Coordinates research priorities with the CEO and team leads.",
        "use_cases": [
            "Commission research on community media trends in 2026",
            "Synthesize findings from multiple research projects",
            "Produce a strategic brief on a new opportunity",
            "Prioritize the research queue for the quarter",
            "Present research findings to support a decision",
        ],
        "integrations": ["All research agents", "Web Search", "Academic Search", "Airtable"],
    },
    "media_platform_researcher": {
        "tagline": "Mapping the digital media platform landscape",
        "expertise": ["Platform Analysis", "Digital Media Trends", "Algorithm Research", "Audience Analytics", "Platform Comparison", "Emerging Platforms"],
        "capabilities": "Researches digital media platforms relevant to Street Voices' mission. Analyzes platform features, audience demographics, and algorithm changes. Compares hosting options for community radio and podcast content. Tracks emerging platforms and technologies in the creator economy. Produces platform assessment reports with recommendations for Street Voices' digital strategy.",
        "use_cases": [
            "Compare podcast hosting platforms for community radio",
            "Research algorithm changes on social media platforms",
            "Analyze audience demographics on a specific platform",
            "Evaluate emerging platforms for content distribution",
            "Produce a platform strategy recommendation report",
        ],
        "integrations": ["Web Search", "Web Fetch", "News Search", "Data analysis tools"],
    },
    "media_program_researcher": {
        "tagline": "Scouting programs, funding, and partnerships",
        "expertise": ["Program Research", "Funding Opportunities", "Partnership Scouting", "Policy Analysis", "Benchmarking", "Best Practices"],
        "capabilities": "Researches community media programs, funding opportunities, and potential partnerships. Scans arts councils, foundations, and government programs for funding aligned with Street Voices' mission. Benchmarks against peer organizations to identify best practices. Analyzes policy developments affecting community media in Canada. Produces opportunity briefs with eligibility requirements and strategic fit assessments.",
        "use_cases": [
            "Scan for new funding opportunities this quarter",
            "Research peer organizations in community media",
            "Analyze a new government policy affecting the sector",
            "Benchmark our programs against similar organizations",
            "Produce an opportunity brief for a potential partnership",
        ],
        "integrations": ["Web Search", "Grants Database", "News Search", "Academic Search"],
    },
    "street_bot_researcher": {
        "tagline": "AI and automation intelligence for nanobot evolution",
        "expertise": ["AI/ML Research", "Agent Framework Analysis", "Tool Discovery", "Automation Patterns", "LLM Evaluation", "MCP Ecosystem"],
        "capabilities": "Researches AI technologies, agent frameworks, and automation patterns to evolve the nanobot platform. Evaluates new LLM models, MCP servers, and tool integrations. Tracks the rapidly evolving AI agent ecosystem for relevant innovations. Tests new capabilities and produces technical evaluation reports. Identifies opportunities to improve nanobot's effectiveness through new technologies.",
        "use_cases": [
            "Evaluate a new LLM model for agent performance",
            "Research new MCP servers for potential integration",
            "Track developments in the AI agent ecosystem",
            "Test a new tool or capability for the platform",
            "Produce a technical evaluation of a new framework",
        ],
        "integrations": ["Web Search", "GitHub", "Academic Search", "News Search"],
    },
    "research_memory": {
        "tagline": "Research archive and knowledge synthesis",
        "expertise": ["Research Archive", "Finding Indexing", "Source Tracking", "Trend History", "Cross-Reference", "Knowledge Gaps"],
        "capabilities": "Maintains the long-term memory of all research activities. Archives findings, sources, and conclusions from past research projects. Indexes data by topic, date, and relevance for fast retrieval. Tracks how trends evolve over time by linking related research. Identifies knowledge gaps and suggests areas needing investigation. Cross-references findings across research domains.",
        "use_cases": [
            "Find past research on a specific topic",
            "Track how a trend has evolved over time",
            "Identify gaps in our research coverage",
            "Cross-reference findings across multiple projects",
            "Surface relevant past findings for a new investigation",
        ],
        "integrations": ["Memory store", "Event history log", "All research agents"],
    },

    # ── Scraping Team ───────────────────────────────────────────────────────
    "scraping_manager": {
        "tagline": "Data extraction strategist and pipeline architect",
        "expertise": ["Scraping Strategy", "Pipeline Design", "Rate Limiting", "Data Quality", "Anti-Detection", "Ethical Scraping"],
        "capabilities": "Leads the scraping team, designing data extraction strategies and managing scraping pipelines. Plans scraping targets, schedules, and rate limits. Ensures ethical scraping practices — respecting robots.txt, rate limits, and terms of service. Validates data quality and handles edge cases. Coordinates with the research team to fulfill data collection needs.",
        "use_cases": [
            "Design a scraping strategy for a new data source",
            "Set up a scheduled scraping pipeline",
            "Review scraping results for data quality issues",
            "Resolve a blocked or rate-limited scraper",
            "Plan ethical data collection for a research project",
        ],
        "integrations": ["Scraping Agent", "Playwright Browser", "Data storage", "Scheduling"],
    },
    "scraping_agent": {
        "tagline": "Precision web scraper and data extractor",
        "expertise": ["Web Scraping", "HTML Parsing", "Data Extraction", "Browser Automation", "API Reverse Engineering", "Data Cleaning"],
        "capabilities": "Executes web scraping tasks using Playwright browser automation and HTML parsing tools. Extracts structured data from websites, APIs, and web applications. Handles dynamic content, JavaScript-rendered pages, and authentication flows. Cleans and structures extracted data for downstream use. Adapts to website changes and anti-bot measures while maintaining ethical practices.",
        "use_cases": [
            "Scrape event listings from a community calendar site",
            "Extract structured data from a government database",
            "Capture content from a dynamic web application",
            "Parse and clean HTML from a batch of web pages",
            "Monitor a website for content changes",
        ],
        "integrations": ["Playwright MCP", "HTML Parser", "Data Extractor", "Web Fetch"],
    },
    "scraper_memory": {
        "tagline": "Scraping patterns and site intelligence",
        "expertise": ["Site Profiles", "Selector History", "Error Patterns", "Rate Limit Tracking", "Data Schema Memory", "Anti-Bot Intelligence"],
        "capabilities": "Maintains the long-term memory of all scraping operations. Stores site profiles with working selectors, rate limit thresholds, and structural patterns. Tracks which approaches work for specific sites and which trigger blocks. Remembers data schemas and transformation rules for recurring scraping targets. Indexes error patterns and their resolutions for faster troubleshooting.",
        "use_cases": [
            "Recall the working selectors for a previously scraped site",
            "Find the rate limit threshold for a specific domain",
            "Surface past errors and fixes for a scraping target",
            "Remember the data schema from a previous extraction",
            "Track which sites have changed structure recently",
        ],
        "integrations": ["Memory store", "Event history log", "All scraping agents"],
    },
}


def nanoid(size: int = 21) -> str:
    """Generate a nanoid-style random string."""
    alphabet = string.ascii_letters + string.digits + "_-"
    return "".join(random.choices(alphabet, k=size))


def generate_agent_id() -> str:
    """Generate a LibreChat-compatible agent ID."""
    return f"agent_{nanoid(21)}"


def load_system_prompt(team_dir: Path, prompt_filename: str) -> str:
    """Load a system prompt .md file for an agent."""
    prompt_path = team_dir / prompt_filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return ""


def load_team_agents(team_dir: Path) -> list[dict]:
    """Load agents from a team's agents.yaml file."""
    yaml_path = team_dir / "agents.yaml"
    if not yaml_path.exists():
        return []

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    agents_raw = data.get("agents", [])
    team_name = team_dir.name

    agents = []
    for agent in agents_raw:
        system_prompt_file = agent.get("system_prompt", "")
        instructions = ""
        if system_prompt_file:
            instructions = load_system_prompt(team_dir, system_prompt_file)

        agents.append({
            "name": agent["name"],
            "team": team_name,
            "role": agent.get("role", "member"),
            "description": agent.get("description", ""),
            "model": agent.get("model", "default"),
            "tools": agent.get("tools", []),
            "handoffs": agent.get("handoffs", []),
            "system_prompt": system_prompt_file,
            "instructions": instructions,
            "max_iterations": agent.get("max_iterations", 10),
        })

    return agents


def format_agent_name(raw_name: str) -> str:
    """Convert snake_case agent name to display name."""
    special = {"ceo": "CEO", "devops": "DevOps", "tts": "TTS"}
    if raw_name in special:
        return special[raw_name]
    return raw_name.replace("_", " ").title()


def build_agent_document(agent: dict, now: datetime) -> dict:
    """Build a LibreChat agent document from nanobot agent data."""
    agent_id = generate_agent_id()
    display_name = format_agent_name(agent["name"])
    category = TEAM_CATEGORY_MAP.get(agent["team"], "general")
    is_lead = agent["role"] == "lead"
    is_memory = agent["role"] == "memory"

    # ── Rich description from AGENT_PROFILES ────────────────────────────
    profile = AGENT_PROFILES.get(agent["name"], {})
    tagline = profile.get("tagline", agent["description"])
    expertise = profile.get("expertise", [])
    capabilities = profile.get("capabilities", "")
    use_cases = profile.get("use_cases", [])
    integrations = profile.get("integrations", [])

    desc_parts = []

    # Tagline + role badge
    if tagline:
        desc_parts.append(tagline)

    desc_parts.append("")  # blank line

    if is_lead:
        desc_parts.append(f"🎯 Team Lead — {CATEGORY_LABELS.get(category, category)}")
    elif is_memory:
        desc_parts.append(f"🧠 Memory Keeper — {CATEGORY_LABELS.get(category, category)}")
    else:
        desc_parts.append(f"📋 {CATEGORY_LABELS.get(category, category)} Team")

    # Expertise tags
    if expertise:
        desc_parts.append(f"🏷️ {' · '.join(expertise)}")

    desc_parts.append("")  # blank line

    # Capabilities paragraph
    if capabilities:
        desc_parts.append(f"**What I do:**\n{capabilities}")
        desc_parts.append("")

    # Use cases
    if use_cases:
        desc_parts.append("**Ask me to:**")
        for uc in use_cases:
            desc_parts.append(f"• {uc}")
        desc_parts.append("")

    # Tools info (from YAML)
    if agent["tools"]:
        tool_names = [TOOL_DISPLAY.get(t, t) for t in agent["tools"]]
        desc_parts.append(f"🔧 **Tools:** {', '.join(tool_names)}")

    # Integrations (from profiles)
    if integrations:
        desc_parts.append(f"🔗 **Integrations:** {', '.join(integrations)}")

    # Handoffs info (from YAML)
    if agent["handoffs"]:
        handoff_names = [format_agent_name(h) for h in agent["handoffs"]]
        desc_parts.append(f"↔️ **Connects to:** {', '.join(handoff_names)}")

    full_description = "\n".join(desc_parts)

    # ── Instructions from system prompt + profile context ───────────────
    if agent["instructions"]:
        instructions = agent["instructions"]
    elif capabilities:
        instructions = (
            f"You are {display_name}, a specialized agent in the "
            f"{CATEGORY_LABELS.get(category, category)} team at Street Voices.\n\n"
            f"{capabilities}\n\n"
            f"Model: {MODEL}"
        )
    else:
        instructions = (
            f"You are {display_name}, a specialized agent in the "
            f"{CATEGORY_LABELS.get(category, category)} team."
        )

    # ── Avatar (MDI icon URL matching librechat.yaml picker icons) ─────
    icon_ref = AGENT_ICON_MAP.get(agent["name"], "robot.svg")
    avatar = {
        "filepath": icon_ref if icon_ref.startswith("http") else f"{MDI_BASE}/{icon_ref}",
        "source": "mdi_cdn",
    }

    doc = {
        "id": agent_id,
        "name": display_name,
        "description": full_description,
        "instructions": instructions,
        "provider": PROVIDER,
        "model": MODEL,
        "author": AUTHOR_OID,
        "authorName": "Joel",
        "avatar": avatar,
        "category": category,
        "is_promoted": is_lead,
        "tools": [],
        "versions": [],
        "createdAt": now,
        "updatedAt": now,
    }

    return doc


def build_acl_entry(agent_doc: dict, now: datetime) -> dict:
    """Build a public ACL entry for an agent."""
    return {
        "principalType": "public",
        "resourceType": "agent",
        "resourceId": agent_doc["_id"],
        "permBits": 1,  # VIEW permission
        "grantedBy": AUTHOR_OID,
        "grantedAt": now,
        "createdAt": now,
        "updatedAt": now,
    }


def build_category_doc(value: str, label: str, order: int, now: datetime) -> dict:
    """Build a custom category document."""
    return {
        "value": value,
        "label": label,
        "description": f"Nanobot {label} agents",
        "order": order,
        "isActive": True,
        "custom": True,
        "createdAt": now,
        "updatedAt": now,
        "__v": 0,
    }


def main():
    dry_run = "--dry-run" in sys.argv

    print("🤖 Nanobot Agent Marketplace Seeder")
    print("=" * 50)
    print(f"MongoDB: {MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}")
    print(f"Teams directory: {TEAMS_DIR}")
    print(f"Model: {MODEL}")
    print(f"Avatars: {MDI_BASE}/")
    print(f"Dry run: {dry_run}")
    print()

    # Connect to MongoDB
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[MONGO_DB]

    # Verify author exists
    user = db.users.find_one({"_id": AUTHOR_OID})
    if not user:
        print(f"❌ Author user not found: {AUTHOR_OID}")
        sys.exit(1)
    print(f"✅ Author: {user.get('name', 'Unknown')} ({user.get('email', 'N/A')})")

    # Load all agents from all teams
    all_agents = []
    team_dirs = sorted(TEAMS_DIR.iterdir())
    for team_dir in team_dirs:
        if not team_dir.is_dir():
            continue
        agents = load_team_agents(team_dir)
        if agents:
            print(f"📁 {team_dir.name}: {len(agents)} agents")
            all_agents.extend(agents)

    print(f"\n📊 Total agents found: {len(all_agents)}")

    if not all_agents:
        print("❌ No agents found!")
        sys.exit(1)

    # Check for existing agents and clean them
    existing_count = db.agents.count_documents({"provider": PROVIDER})
    if existing_count > 0:
        print(f"\n🗑️  Removing {existing_count} existing Nanobot agents...")
        if not dry_run:
            existing_ids = [
                doc["_id"]
                for doc in db.agents.find({"provider": PROVIDER}, {"_id": 1})
            ]
            # Clean up ACL entries (handle both casing variants)
            db.aclentries.delete_many({
                "resourceType": {"$in": ["AGENT", "agent"]},
                "resourceId": {"$in": existing_ids},
            })
            db.agents.delete_many({"provider": PROVIDER})
            print("   Done.")

    # Create custom categories if they don't exist
    print("\n📂 Ensuring categories exist...")
    existing_cats = {doc["value"] for doc in db.agentcategories.find({}, {"value": 1})}
    now = datetime.now(timezone.utc)
    new_cats = []
    for i, (value, label) in enumerate(CATEGORY_LABELS.items()):
        if value not in existing_cats:
            cat_doc = build_category_doc(value, label, 10 + i, now)
            new_cats.append(cat_doc)
            print(f"   + {value}: {label}")

    if new_cats and not dry_run:
        db.agentcategories.insert_many(new_cats)
        print(f"   Inserted {len(new_cats)} new categories.")
    elif not new_cats:
        print("   All categories already exist.")

    # Build and insert agent documents
    print("\n🔨 Building agent documents...")
    agent_docs = []
    profiled = 0
    for agent in all_agents:
        doc = build_agent_document(agent, now)
        agent_docs.append(doc)
        if agent["name"] in AGENT_PROFILES:
            profiled += 1

    print(f"   {profiled}/{len(agent_docs)} agents have rich profiles")

    if dry_run:
        print("\n📋 DRY RUN — Would insert these agents:")
        for doc in agent_docs:
            promoted = "⭐" if doc["is_promoted"] else "  "
            has_avatar = "🖼️" if doc.get("avatar") else "  "
            print(f"   {promoted}{has_avatar} {doc['name']:<30} [{doc['category']}] {doc['id']}")
        print(f"\n   Total: {len(agent_docs)} agents")
        print("   No changes made.")
        return

    # Insert agents
    print(f"\n💾 Inserting {len(agent_docs)} agents...")
    result = db.agents.insert_many(agent_docs)
    print(f"   Inserted {len(result.inserted_ids)} agents.")

    # Create public ACL entries for all agents
    print("\n🔓 Creating public ACL entries...")
    acl_entries = []
    for doc in agent_docs:
        acl = build_acl_entry(doc, now)
        acl["resourceId"] = doc["_id"]
        acl_entries.append(acl)

    acl_result = db.aclentries.insert_many(acl_entries)
    print(f"   Created {len(acl_result.inserted_ids)} ACL entries.")

    # Summary
    print("\n" + "=" * 50)
    print("✅ Marketplace seeded successfully!")
    print()

    # Print agents grouped by team
    by_team = {}
    for doc in agent_docs:
        team = doc["category"]
        by_team.setdefault(team, []).append(doc)

    for team, docs in sorted(by_team.items()):
        label = CATEGORY_LABELS.get(team, team)
        print(f"\n  {label}:")
        for doc in docs:
            promoted = "⭐" if doc["is_promoted"] else "  "
            print(f"    {promoted} {doc['name']}")

    print(f"\n  Total: {len(agent_docs)} agents across {len(by_team)} teams")
    print(f"  Model: {MODEL}")
    print(f"  Avatars: {MDI_BASE}/")
    print(f"  Public ACL entries: {len(acl_entries)}")
    print(f"  Rich profiles: {profiled}/{len(agent_docs)}")
    print(f"\n  View at: http://localhost:3180/agents")


if __name__ == "__main__":
    main()
