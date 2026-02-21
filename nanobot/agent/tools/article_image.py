"""Agent tool for generating Canva-style article images."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class ArticleImageTool(Tool):
    """Generate cover + body article images from headline, body text, and a hero photo."""

    def __init__(self, base_url: str = "http://localhost:18790") -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "generate_article_image"

    @property
    def description(self) -> str:
        return (
            "Generate a set of on-brand Street Voices article images (cover page + body page) "
            "in 1080×1350px format, matching the Canva article template. "
            "Takes a headline, body text, hero image URL, and optional category tag. "
            "Returns URLs to the generated cover and body PNG images. "
            "Use this instead of Canva when generating article/news post images."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "The article headline text (displayed on the cover image).",
                },
                "body_text": {
                    "type": "string",
                    "description": (
                        "The article body text for the second page. "
                        "Use \\n for line breaks. Lines that are short and title-case "
                        "are auto-detected as subheadings and rendered in bold."
                    ),
                },
                "hero_image_url": {
                    "type": "string",
                    "description": "URL to the hero/cover photo (will be crop-to-fill).",
                },
                "category": {
                    "type": "string",
                    "description": "Category label on the yellow banner (default: ARTICLE).",
                    "default": "ARTICLE",
                },
            },
            "required": ["headline", "body_text", "hero_image_url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from nanobot.services.article_image import generate_article_images

        headline = kwargs["headline"]
        body_text = kwargs["body_text"]
        hero_url = kwargs["hero_image_url"]
        category = kwargs.get("category", "ARTICLE")

        try:
            result = await generate_article_images(
                headline=headline,
                body_text=body_text,
                hero_image_url=hero_url,
                category=category,
            )
        except Exception as e:
            logger.exception("Article image generation failed")
            return f"Error generating article images: {e}"

        cover_url = f"{self._base_url}{result['cover_url']}"
        body_url = f"{self._base_url}{result['body_url']}"

        return (
            f"Article images generated successfully!\n\n"
            f"**Cover image:** {cover_url}\n"
            f"**Body image:** {body_url}\n\n"
            f"Image ID: {result['image_id']}"
        )
