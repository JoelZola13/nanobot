"""Autonomous browser agent tool using Playwright MCP primitives."""

import json
import re
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider

_STALE_PATTERNS = re.compile(
    r"(Target closed|frame was detached|stale|element is not attached|execution context was destroyed)",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """\
You are a browser automation agent. You control a Chromium browser via structured actions.

You will receive:
1. The user's GOAL
2. A text snapshot of the page (accessibility tree with element refs like [ref1], [ref2])
3. Optionally a screenshot of the current page

Your job: decide the SINGLE next action to take toward completing the goal.

Respond with EXACTLY one JSON object (no markdown, no explanation):
{
  "reasoning": "brief explanation of what you see and why you're taking this action",
  "action": "click|type|navigate|scroll|snapshot|done|wait",
  "params": { ... },
  "done": false
}

Actions:
- click: {"ref": "ref_123"} — click an element by its ref ID from the snapshot
- type: {"ref": "ref_123", "text": "hello"} — type text into an input element
- navigate: {"url": "https://..."} — navigate to a URL
- scroll: {"direction": "down", "amount": 3} — scroll the page
- snapshot: {} — take a fresh page snapshot (useful after actions that change the page)
- wait: {"seconds": 2} — wait for page to load
- done: {"summary": "what was accomplished"} — goal is complete

Rules:
- Take ONE action at a time
- After click/type/navigate, the next iteration will give you a fresh snapshot
- Use element refs from the snapshot (e.g. "ref_123"), not CSS selectors
- If a page needs to load, use wait then snapshot
- If stuck after 3 attempts on the same step, try an alternative approach
- When the goal is achieved, use action "done" with a summary
"""


class ComputerUseManager:
    """Orchestrates autonomous browser tasks via screenshot→reason→act loop."""

    def __init__(
        self,
        registry: ToolRegistry,
        provider: LLMProvider,
        model: str,
        vision_model: str | None = None,
    ):
        self.registry = registry
        self.provider = provider
        self.model = model
        self.vision_model = vision_model or model

    def _find_mcp_tool(self, original_name: str) -> str | None:
        """Find the full MCP tool name for a Playwright browser tool."""
        for name in self.registry.tool_names:
            if name.endswith(f"_{original_name}"):
                return name
        return None

    async def _call_mcp(self, original_name: str, **kwargs: Any) -> str:
        """Call a Playwright MCP tool by its original name."""
        full_name = self._find_mcp_tool(original_name)
        if not full_name:
            return f"Error: MCP tool '{original_name}' not found"
        result = await self.registry.execute(full_name, kwargs)

        # Stale DOM retry
        if _STALE_PATTERNS.search(result) and original_name not in ("browser_snapshot", "browser_navigate"):
            logger.warning(f"[computer_use] Stale DOM for {original_name}, retrying after snapshot...")
            await self.registry.execute(self._find_mcp_tool("browser_snapshot") or "", {})
            result = await self.registry.execute(full_name, kwargs)
        return result

    async def _get_screenshot_data_url(self) -> str | None:
        """Take a screenshot and return the base64 data URL."""
        tool_name = self._find_mcp_tool("browser_take_screenshot")
        if not tool_name:
            return None
        tool_obj = self.registry.get(tool_name)
        if not tool_obj:
            return None

        # Clear any pending images, execute, then grab the data URL
        if hasattr(tool_obj, "_pending_images"):
            tool_obj._pending_images.clear()

        await self.registry.execute(tool_name, {})

        if hasattr(tool_obj, "_pending_images") and tool_obj._pending_images:
            data_url = tool_obj._pending_images[0]
            tool_obj._pending_images.clear()
            return data_url
        return None

    async def execute_goal(
        self,
        goal: str,
        url: str | None = None,
        max_steps: int = 30,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Execute a high-level browser goal autonomously.

        Returns a summary of what was accomplished.
        """
        if url:
            nav_result = await self._call_mcp("browser_navigate", url=url)
            if on_progress:
                await on_progress(f"Navigated to {url}")
            logger.info(f"[computer_use] Navigated to {url}: {nav_result[:100]}")

        history: list[str] = []

        for step in range(max_steps):
            # 1. Take snapshot (a11y tree)
            snapshot = await self._call_mcp("browser_snapshot")

            # 2. Take screenshot for vision
            screenshot_url = await self._get_screenshot_data_url()

            # 3. Build messages for inner LLM
            user_content: list[dict[str, Any]] = []
            user_content.append({
                "type": "text",
                "text": (
                    f"GOAL: {goal}\n\n"
                    f"STEP: {step + 1}/{max_steps}\n\n"
                    f"PREVIOUS ACTIONS:\n"
                    + ("\n".join(f"- {h}" for h in history[-5:]) if history else "(none)")
                    + f"\n\nPAGE SNAPSHOT:\n{snapshot[:8000]}"
                ),
            })
            if screenshot_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": screenshot_url},
                })

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # 4. Ask vision LLM for next action
            try:
                response = await self.provider.chat(
                    messages=messages,
                    model=self.vision_model,
                    max_tokens=1024,
                    temperature=0.1,
                )
            except Exception as e:
                logger.error(f"[computer_use] LLM call failed at step {step + 1}: {e}")
                return f"Error: LLM call failed at step {step + 1}: {e}"

            raw = (response.content or "").strip()
            logger.info(f"[computer_use] Step {step + 1} LLM response: {raw[:200]}")

            # 5. Parse JSON action
            try:
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                action_data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"[computer_use] Failed to parse LLM response as JSON: {raw[:200]}")
                history.append(f"Step {step + 1}: LLM returned invalid JSON, retrying")
                continue

            action = action_data.get("action", "")
            params = action_data.get("params", {})
            reasoning = action_data.get("reasoning", "")
            is_done = action_data.get("done", False)

            progress_msg = f"Step {step + 1}: {action} — {reasoning[:80]}"
            history.append(progress_msg)
            if on_progress:
                await on_progress(progress_msg)

            # 6. Execute action
            if action == "done" or is_done:
                summary = params.get("summary", reasoning)
                logger.info(f"[computer_use] Goal completed in {step + 1} steps: {summary}")
                return f"Browser task completed in {step + 1} steps.\n\nSummary: {summary}"

            elif action == "click":
                ref = params.get("ref", "")
                result = await self._call_mcp("browser_click", element=ref)
                logger.debug(f"[computer_use] click {ref}: {result[:100]}")

            elif action == "type":
                ref = params.get("ref", "")
                text = params.get("text", "")
                # Try fill first (for inputs), fall back to type
                result = await self._call_mcp("browser_type", element=ref, text=text)
                logger.debug(f"[computer_use] type into {ref}: {result[:100]}")

            elif action == "navigate":
                nav_url = params.get("url", "")
                result = await self._call_mcp("browser_navigate", url=nav_url)
                logger.debug(f"[computer_use] navigate to {nav_url}: {result[:100]}")

            elif action == "scroll":
                direction = params.get("direction", "down")
                # Map to Playwright scroll — use keyboard shortcuts as fallback
                if direction == "down":
                    result = await self._call_mcp("browser_press_key", key="PageDown")
                elif direction == "up":
                    result = await self._call_mcp("browser_press_key", key="PageUp")
                else:
                    result = await self._call_mcp("browser_press_key", key="PageDown")
                logger.debug(f"[computer_use] scroll {direction}: {result[:100]}")

            elif action == "wait":
                import asyncio
                seconds = min(params.get("seconds", 2), 10)
                await asyncio.sleep(seconds)
                logger.debug(f"[computer_use] waited {seconds}s")

            elif action == "snapshot":
                # Just continue — snapshot happens at top of loop
                pass

            else:
                logger.warning(f"[computer_use] Unknown action: {action}")
                history.append(f"Unknown action: {action}")

        return f"Browser task reached step limit ({max_steps}). Last actions:\n" + "\n".join(history[-5:])


class ComputerUseTool(Tool):
    """Autonomous browser agent — give it a goal and optional URL."""

    def __init__(self, manager: ComputerUseManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "computer_use"

    @property
    def description(self) -> str:
        return (
            "Autonomously control a browser to accomplish a goal. Provide a high-level "
            "task description and optionally a starting URL. The tool will navigate, click, "
            "type, and scroll as needed to complete the task. Uses vision to understand "
            "page content. Examples: 'Search Google for X and summarize the first result', "
            "'Go to github.com/anthropics and find the latest release', "
            "'Fill out the contact form on example.com with name John Doe'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "High-level description of what to accomplish in the browser.",
                },
                "url": {
                    "type": "string",
                    "description": "Starting URL to navigate to (optional — can also navigate during execution).",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of actions to take (default 30).",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["goal"],
        }

    async def execute(self, **kwargs: Any) -> str:
        goal = kwargs.get("goal", "")
        url = kwargs.get("url")
        max_steps = kwargs.get("max_steps", 30)

        if not goal:
            return "Error: goal is required"

        return await self._manager.execute_goal(goal=goal, url=url, max_steps=max_steps)
