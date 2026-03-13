"""Grants database tool — searches Grants.gov (US) and Canadian funding sources."""

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


# Canadian arts/culture funders with known programs and deadlines
CANADIAN_FUNDERS = [
    {
        "name": "Canada Council for the Arts",
        "url": "https://canadacouncil.ca/funding",
        "programs": [
            "Explore and Create", "Creating, Knowing and Sharing: Indigenous Arts",
            "Digital Strategy Fund", "Arts Abroad", "Concept to Realization",
        ],
        "focus": "arts, media, digital, literary, performing arts, visual arts, Indigenous arts",
        "typical_award": "$5,000 - $350,000",
        "note": "Primary federal arts funder. Multiple deadlines per year per program.",
    },
    {
        "name": "Ontario Arts Council (OAC)",
        "url": "https://www.arts.on.ca/grants",
        "programs": [
            "Projects: Media Arts", "Projects: Multi and Inter-Arts",
            "Community and Multidisciplinary Projects",
        ],
        "focus": "arts, media, community arts, Ontario-based",
        "typical_award": "$5,000 - $50,000",
        "note": "Ontario provincial arts funder. Must be Ontario-based.",
    },
    {
        "name": "Toronto Arts Council (TAC)",
        "url": "https://torontoartscouncil.org/grants",
        "programs": [
            "TAC Grants to Organizations", "TAC Grants to Artists",
            "Community Arts Projects",
        ],
        "focus": "arts, community, Toronto-based",
        "typical_award": "$2,000 - $30,000",
        "note": "Municipal funder. Must be Toronto-based.",
    },
    {
        "name": "SSHRC (Social Sciences and Humanities Research Council)",
        "url": "https://www.sshrc-crsh.gc.ca/funding-financement/programs-programmes-eng.aspx",
        "programs": [
            "Insight Grants", "Connection Grants", "Partnership Grants",
            "Partnership Development Grants",
        ],
        "focus": "research, social sciences, humanities, community-engaged research",
        "typical_award": "$7,000 - $2,500,000",
        "note": "Federal research funder. Usually requires academic partnership.",
    },
    {
        "name": "Canadian Heritage",
        "url": "https://www.canada.ca/en/canadian-heritage/services/funding.html",
        "programs": [
            "Canada Arts Presentation Fund", "Building Communities Through Arts and Heritage",
            "Canada Cultural Spaces Fund", "Canada Music Fund",
        ],
        "focus": "heritage, culture, festivals, performing arts, music, community events",
        "typical_award": "$10,000 - $500,000",
        "note": "Federal cultural funder. Various programs with different eligibility.",
    },
    {
        "name": "Ontario Trillium Foundation",
        "url": "https://otf.ca/what-we-fund",
        "programs": ["Seed Grant", "Grow Grant", "Resilient Communities Fund"],
        "focus": "community development, youth, environment, poverty reduction, active people",
        "typical_award": "$5,000 - $150,000",
        "note": "Ontario's largest granting foundation. Broad community focus.",
    },
    {
        "name": "Community Foundations of Canada / Toronto Foundation",
        "url": "https://torontofoundation.ca/grants",
        "programs": ["Community Grants", "Vital Signs", "Youth Challenge Fund"],
        "focus": "community development, equity, youth, civic engagement",
        "typical_award": "$5,000 - $50,000",
        "note": "Place-based philanthropy. Check specific community foundation for local programs.",
    },
]


class GrantsDatabaseTool(Tool):
    """Search grant opportunities via Grants.gov API and Canadian funding databases."""

    @property
    def name(self) -> str:
        return "grants_database"

    @property
    def description(self) -> str:
        return (
            "Search grant opportunity databases. Covers Grants.gov (US federal) "
            "and Canadian funders (Canada Council, OAC, SSHRC, Canadian Heritage, "
            "Ontario Trillium, Toronto Arts Council, community foundations). "
            "Search by keyword, category, agency, or country. Returns grant titles, "
            "agencies, deadlines, and award amounts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for grant opportunities",
                },
                "category": {
                    "type": "string",
                    "description": "Grant category filter (e.g., 'arts', 'health', 'education', 'community', 'research')",
                },
                "agency": {
                    "type": "string",
                    "description": "Funding agency filter (e.g., 'NIH', 'NSF', 'Canada Council', 'OAC')",
                },
                "country": {
                    "type": "string",
                    "enum": ["all", "us", "canada"],
                    "description": "Search US federal (Grants.gov), Canadian funders, or both. Default: all",
                },
                "min_amount": {
                    "type": "integer",
                    "description": "Minimum award amount filter",
                },
                "max_amount": {
                    "type": "integer",
                    "description": "Maximum award amount filter",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        category = kwargs.get("category", "").strip()
        agency = kwargs.get("agency", "").strip()
        country = kwargs.get("country", "all").strip().lower()
        min_amount = kwargs.get("min_amount")
        max_amount = kwargs.get("max_amount")

        if not query:
            return "Error: No search query provided."

        results: list[str] = []
        query_lower = query.lower()

        # Search Canadian funders (local database)
        if country in ("all", "canada"):
            canadian_results = self._search_canadian(query_lower, category, agency)
            if canadian_results:
                results.append("## Canadian Funding Sources\n")
                results.extend(canadian_results)

        # Search Grants.gov (US federal)
        if country in ("all", "us"):
            us_results = await self._search_grants_gov(query, category, agency, min_amount, max_amount)
            if us_results:
                results.append("\n## US Federal Grants (Grants.gov)\n")
                results.append(us_results)

        if not results:
            return f"No grant opportunities found for: {query}"

        return "\n".join(results)

    def _search_canadian(self, query: str, category: str, agency: str) -> list[str]:
        """Search the Canadian funders database by keyword matching."""
        matches: list[str] = []
        query_terms = query.lower().split()

        for funder in CANADIAN_FUNDERS:
            # Filter by specific agency name if provided
            if agency and agency.lower() not in funder["name"].lower():
                continue

            # Check if query terms match funder focus areas, programs, or name
            funder_text = (
                funder["name"] + " " + funder["focus"] + " " +
                " ".join(funder["programs"])
            ).lower()

            # Score by number of matching terms
            score = sum(1 for term in query_terms if term in funder_text)
            if score == 0:
                continue

            # Filter by category if provided
            if category and category.lower() not in funder["focus"].lower():
                continue

            lines = [f"**{funder['name']}**"]
            lines.append(f"  Website: {funder['url']}")
            lines.append(f"  Focus: {funder['focus']}")
            lines.append(f"  Typical Awards: {funder['typical_award']}")
            lines.append(f"  Programs: {', '.join(funder['programs'])}")
            if funder.get("note"):
                lines.append(f"  Note: {funder['note']}")
            lines.append("")
            matches.append("\n".join(lines))

        return matches

    async def _search_grants_gov(
        self, query: str, category: str, agency: str,
        min_amount: int | None, max_amount: int | None,
    ) -> str:
        """Search Grants.gov API for US federal opportunities."""
        search_params: dict[str, Any] = {
            "keyword": query,
            "oppStatuses": "forecasted|posted",
        }
        if agency:
            search_params["agencies"] = agency
        if min_amount is not None:
            search_params["awardFloor"] = min_amount
        if max_amount is not None:
            search_params["awardCeiling"] = max_amount

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
                return f"Grants.gov API returned HTTP {e.response.status_code}. The API may be temporarily unavailable."
        except httpx.RequestError as e:
            return f"Request failed: {e}"

        # Parse response
        opportunities = data.get("oppHits", data.get("opportunities", data.get("data", [])))

        if not opportunities:
            return ""

        lines: list[str] = []
        for i, opp in enumerate(opportunities[:15], 1):
            title = opp.get("title", opp.get("oppTitle", "Untitled"))
            opp_agency = opp.get("agency", opp.get("agencyName", ""))
            close_date = opp.get("closeDate", opp.get("closingDate", ""))
            award = opp.get("awardCeiling", opp.get("award", ""))
            opp_number = opp.get("number", opp.get("oppNumber", ""))
            opp_status = opp.get("status", opp.get("oppStatus", ""))
            desc = opp.get("description", opp.get("synopsis", ""))

            lines.append(f"{i}. **{title}**")
            if opp_number:
                lines.append(f"   Number: {opp_number}")
            if opp_agency:
                lines.append(f"   Agency: {opp_agency}")
            if opp_status:
                lines.append(f"   Status: {opp_status}")
            if close_date:
                lines.append(f"   Deadline: {close_date}")
            if award:
                if isinstance(award, (int, float)):
                    lines.append(f"   Award Ceiling: ${int(award):,}")
                else:
                    lines.append(f"   Award: {award}")
            if desc:
                clean_desc = str(desc)[:300].replace("\n", " ").strip()
                lines.append(f"   {clean_desc}{'...' if len(str(desc)) > 300 else ''}")
            lines.append("")

        return "\n".join(lines)
