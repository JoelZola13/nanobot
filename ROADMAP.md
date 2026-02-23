# Nanobot-Based OpenClaw-Style Assistant — Build Plan

## North Star
Build a personal AI assistant that:
1) understands intent across chat + voice
2) plans multi-step tasks reliably
3) acts through tools safely (permissioned + auditable)
4) remembers what matters (scoped + correct)
5) explains actions and stays controllable

## Guiding Principles
- Secure-by-default (treat all inbound content as untrusted)
- Local-first data ownership where possible
- Minimal stable core (“agent kernel”), everything else is plugins/skills
- Observable + testable (no silent failures)
- User control over autonomy (clear consent gates)

---

## Architecture Overview

### Core Loop (Kernel)
- Inbound message event
- Policy checks (identity/allowlist, safety, permissions)
- Plan (task decomposition)
- Execute (tool calls)
- Observe + update memory
- Respond (with progress + citations)

### Components
- Channels: Telegram/Discord/Slack/Email/etc.
- Agent Runtime: planner + executor + recovery
- Tool Layer: local tools + MCP tool servers
- Skill Layer: versioned packages with manifests
- Memory: profile + project + episodic
- Policy Engine: permissions, confirmations, secrets redaction
- Observability: logs, traces, audit trail, replay

---

## Roadmap

### P0 — Safety & Control Plane (Must-have)
- [ ] Threat model (assets, trust boundaries, attacker goals)
- [ ] Default allowlist/pairing for DMs + channel senders
- [ ] Secret store (OS keychain or encrypted vault)
- [ ] Redaction (logs, prompts, tool outputs)
- [ ] Permission scopes for every tool/skill
- [ ] Confirmations for irreversible actions (send/delete/pay)
- [ ] Tool sandboxing (restricted FS + network allowlists)
- [ ] Audit log (request → plan → tool calls → actions) + replay

### P1 — Plugins & Skills (Ecosystem)
- [ ] Plugin SDK (BaseChannel, BaseTool, Provider interfaces)
- [ ] Skill manifest spec (permissions, secrets, entrypoint, tests)
- [ ] Skill registry support (install/pin/verify)
- [ ] MCP-first external tools (connect/disconnect per workspace)
- [ ] Skill security: signing (optional), warnings, scanning

### P2 — Memory That Helps
- [ ] Profile memory (preferences, constraints, tone)
- [ ] Project memory (scoped notes + docs + artifacts)
- [ ] Episodic memory (summaries + “what changed”)
- [ ] Retrieval policy (deterministic first, vector second)
- [ ] Memory citations surfaced to user

### P3 — Reliability & Long-Running Work
- [ ] Workflow engine with checkpoints + resume
- [ ] Job queue for scheduled/async tasks
- [ ] Idempotency keys for side-effect tools
- [ ] Retries/backoff with structured error classes
- [ ] Progress streaming for multi-step execution

### P4 — Evaluation & Quality
- [ ] Golden task suite (real tasks)
- [ ] Unit tests for skills/tools
- [ ] Prompt-injection regression suite
- [ ] Metrics dashboard (success rate, tool errors, cost, latency)

---

## Skill Manifest (Draft Spec)

Example `skill.yml`:

name: "inbox-cleaner"
version: "0.1.0"
description: "Triage email with user-defined rules"
entrypoint: "inbox_cleaner:run"

permissions:
  - email.read
  - email.write   # requires confirmation for send/archive/delete

secrets:
  - gmail_oauth_token

confirmations:
  - action: "email.delete"
    prompt: "Delete {count} messages from {sender}? (y/N)"
  - action: "email.send"
    prompt: "Send this email to {to}? (y/N)"

timeouts:
  default_seconds: 30

tests:
  command: "pytest -q"

---

## Security Model

### Trust Levels
- Trusted: user/system instructions, local policy config
- Semi-trusted: tool outputs (still validate)
- Untrusted: web pages, inbound emails, inbound chat messages, attachments

### Rules
- Untrusted content cannot directly trigger tool calls
- Any write/delete/send/pay requires explicit scope + (often) confirmation
- Secrets never enter prompts or logs
- External network calls are allowlisted and visible in audit log

---

## Definition of Done (v1)
- [ ] Can complete 20/20 core tasks in golden suite
- [ ] No critical prompt-injection regressions in test suite
- [ ] All actions are auditable + replayable
- [ ] “Read-only mode” is usable as a safe default
- [ ] Onboarding is < 10 minutes for a new machine
