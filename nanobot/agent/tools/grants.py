"""Grants database tool using Grants.gov API."""

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class GrantsDatabaseTool(Tool):
    """Search grant opportunities via Grants.gov API."""

    @property
    def name(self) -> str:
        return "grants_database"

    @property
    def description(self) -> str:
        return (
            "Search and query grant opportunity databases (Grants.gov). "
            "Search by keyword, category, or agency. Returns grant titles, "
            "agencies, deadlines, and award amounts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for grant opportunities"},
                "category": {
                    "type": "string",
                    "description": "Grant category filter (e.g., 'health', 'education', 'science')",
                },
                "agency": {
                    "type": "string",
                    "description": "Funding agency filter (e.g., 'NIH', 'NSF', 'DOE')",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        category = kwargs.get("category", "").strip()
        agency = kwargs.get("agency", "").strip()

        if not query:
            return "Error: No search query provided."

        # Build search request
        search_params: dict[str, Any] = {
            "keyword": query,
            "oppStatuses": "forecasted|posted",
        }
        if agency:
            search_params["agencies"] = agency

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://api.grants.gov/v1/api/search",
                    json=search_params,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            # Fallback: try the v2 endpoint
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(
                        "https://api.grants.gov/search/opportunities",
                        params={"keyword": query, "status": "posted"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception:
                return f"Error: Grants.gov API returned HTTP {e.response.status_code}. The API may be temporarily unavailable."
        except httpx.RequestError as e:
            return f"Error: Request failed: {e}"

        # Parse response (Grants.gov has varying response formats)
        opportunities = data.get("oppHits", data.get("opportunities", data.get("data", [])))

        if not opportunities:
            return f"No grant opportunities found for: {query}"

        lines = [f"Grant opportunities for: {query}\n"]
        for i, opp in enumerate(opportunities[:15], 1):
            title = opp.get("title", opp.get("oppTitle", "Untitled"))
            opp_agency = opp.get("agency", opp.get("agencyName", ""))
            close_date = opp.get("closeDate", opp.get("closingDate", ""))
            award = opp.get("awardCeiling", opp.get("award", ""))
            opp_number = opp.get("number", opp.get("oppNumber", ""))
            opp_status = opp.get("status", opp.get("oppStatus", ""))
            desc = opp.get("description", opp.get("synopsis", ""))

            lines.append(f"{i}. {title}")
            if opp_number:
                lines.append(f"   Number: {opp_number}")
            if opp_agency:
                lines.append(f"   Agency: {opp_agency}")
            if opp_status:
                lines.append(f"   Status: {opp_status}")
            if close_date:
                lines.append(f"   Deadline: {close_date}")
            if award:
                lines.append(f"   Award Ceiling: ${int(award):,}" if isinstance(award, (int, float)) else f"   Award: {award}")
            if desc:
                clean_desc = desc[:300].replace('\n', ' ').strip()
                lines.append(f"   {clean_desc}{'...' if len(str(desc)) > 300 else ''}")
            lines.append("")

        return "\n".join(lines)
