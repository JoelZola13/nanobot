"""Migrate existing nanobot MEMORY.md into universal shared memory.

Parses the existing MEMORY.md file and distributes content into
the appropriate universal memory categories (shared, contacts,
decisions, projects).

Usage:
    python -m nanobot.harness.migrate_memory
"""

from __future__ import annotations

import re
from pathlib import Path

from nanobot.harness.memory import UniversalMemory


def migrate(workspace: Path | None = None) -> None:
    """Migrate existing MEMORY.md content to universal memory."""
    workspace = workspace or Path.home() / ".nanobot" / "workspace"
    memory_file = workspace / "memory" / "MEMORY.md"

    if not memory_file.exists():
        print("No existing MEMORY.md found, nothing to migrate.")
        return

    content = memory_file.read_text(encoding="utf-8")
    mem = UniversalMemory(workspace)

    # Parse sections from existing MEMORY.md
    preferences = []
    projects = []
    brand_kits = []
    tools = []
    location = []
    other = []

    current_section = "other"
    for line in content.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Detect section headers
        if line_stripped.lower().startswith("preferences"):
            current_section = "preferences"
            continue
        elif line_stripped.lower().startswith("projects"):
            current_section = "projects"
            continue
        elif line_stripped.lower().startswith("brand kits"):
            current_section = "brand"
            continue
        elif line_stripped.lower().startswith("tools"):
            current_section = "tools"
            continue
        elif line_stripped.lower().startswith("location"):
            current_section = "location"
            continue

        # Strip leading "- " from list items
        entry = re.sub(r"^-\s*", "", line_stripped)
        if not entry:
            continue

        if current_section == "preferences":
            preferences.append(entry)
        elif current_section == "projects":
            projects.append(entry)
        elif current_section == "brand":
            brand_kits.append(entry)
        elif current_section == "tools":
            tools.append(entry)
        elif current_section == "location":
            location.append(entry)
        else:
            other.append(entry)

    # Write to universal memory
    if preferences:
        decisions_content = "# Decisions & Preferences\n\n"
        decisions_content += "## User Preferences (migrated from MEMORY.md)\n\n"
        for p in preferences:
            decisions_content += f"- {p}\n"
        mem.update_decisions(decisions_content)
        print(f"Migrated {len(preferences)} preferences to decisions")

    if projects:
        projects_content = "# Active Projects\n\n"
        projects_content += "## Projects (migrated from MEMORY.md)\n\n"
        for p in projects:
            projects_content += f"- {p}\n"
        mem.update_projects(projects_content)
        print(f"Migrated {len(projects)} project entries")

    # Shared gets brand, tools, location, and other
    shared_parts = []
    if brand_kits:
        shared_parts.append("## Brand Kits\n" + "\n".join(f"- {b}" for b in brand_kits))
    if tools:
        shared_parts.append("## Tool Config\n" + "\n".join(f"- {t}" for t in tools))
    if location:
        shared_parts.append("## Location Context\n" + "\n".join(f"- {l}" for l in location))
    if other:
        shared_parts.append("## Other Context\n" + "\n".join(f"- {o}" for o in other))

    if shared_parts:
        shared_content = "# Universal Shared Memory\n\n" + "\n\n".join(shared_parts) + "\n"
        mem.update_shared(shared_content)
        print(f"Migrated {len(brand_kits) + len(tools) + len(location) + len(other)} entries to shared")

    # Seed contacts with Joel's info
    mem.append_contact("Joel", "Founder of Street Voices, based in Toronto (Little Portugal). Email: joel@streetvoices.ca. Prefers casual communication.")
    print("Seeded Joel contact")

    print(f"\nMigration complete! Universal memory at: {mem.base_dir}")
    print(f"  Shared: {len(mem.get_shared_context())} chars")
    print(f"  Contacts: {len(mem.get_contacts())} chars")
    print(f"  Decisions: {len(mem.get_decisions())} chars")
    print(f"  Projects: {len(mem.get_projects())} chars")


if __name__ == "__main__":
    migrate()
