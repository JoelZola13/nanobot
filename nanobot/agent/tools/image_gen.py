"""Qwen-Image local image generation tool."""

import json
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

QWEN_IMAGE_URL = "http://localhost:18791"


class QwenImageGenTool(Tool):
    """Generate images using a local Qwen-Image-2512 model."""

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "Generate a single high-quality image from a text prompt using the local "
            "Qwen-Image-2512 model. Returns a URL to the generated image. "
            "IMPORTANT: Only call this tool ONCE per user request unless the user "
            "explicitly asks for multiple images. Each call takes ~60 seconds. "
            "Supports aspect ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate. Be detailed for best results.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio of the output image.",
                    "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    "default": "1:1",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Things to exclude from the image (optional).",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt", "")
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        negative_prompt = kwargs.get("negative_prompt", " ")

        if not prompt:
            return "Error: prompt is required"

        body = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": negative_prompt,
            # Server defaults to Lightning (4 steps, cfg 1.0) or Base (20 steps, cfg 4.0)
            # Omit these to use server defaults
        }

        try:
            async with httpx.AsyncClient(timeout=1200.0) as client:
                response = await client.post(f"{QWEN_IMAGE_URL}/generate", json=body)
                if response.status_code != 200:
                    return f"Error: Image generation failed ({response.status_code}): {response.text}"
                data = response.json()
                url = data.get("url", "")
                elapsed = data.get("elapsed_seconds", "?")
                filepath = data.get("file", "")
                logger.info(f"Image generated in {elapsed}s: {filepath}")
                return (
                    f"Image generated successfully in {elapsed}s.\n"
                    f"Show it to the user with exactly this markdown (do NOT duplicate it):\n"
                    f"![image]({url})"
                )
        except httpx.ConnectError:
            return (
                "Error: Qwen-Image server is not running. "
                "Start it with: cd ~/Projects/Qwen-Image && .venv/bin/python server.py"
            )
        except Exception as e:
            return f"Error generating image: {e}"
