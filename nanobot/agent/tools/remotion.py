"""Remotion video tools: compose TSX compositions and render to mp4/gif."""

import asyncio
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class RemotionComposeTool(Tool):
    """Write a new Remotion TSX composition and register it in Root.tsx."""

    name = "remotion_compose"
    description = (
        "Create a new Remotion video composition. Writes a TSX file and registers it in Root.tsx. "
        "The code should export a named React component using Remotion APIs "
        "(useCurrentFrame, useVideoConfig, Sequence, AbsoluteFill, Img, Audio, etc). "
        "After composing, use remotion_render to render it to mp4 or gif."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "PascalCase component/composition name (e.g. 'MyVideo')",
            },
            "code": {
                "type": "string",
                "description": "Full TSX source code for the composition file. Must export a named component matching the name parameter.",
            },
            "duration_frames": {
                "type": "integer",
                "description": "Duration in frames (e.g. 150 = 5 seconds at 30fps)",
                "minimum": 1,
            },
            "fps": {
                "type": "integer",
                "description": "Frames per second (default 30)",
                "minimum": 1,
            },
            "width": {
                "type": "integer",
                "description": "Video width in pixels (default 1920)",
                "minimum": 1,
            },
            "height": {
                "type": "integer",
                "description": "Video height in pixels (default 1080)",
                "minimum": 1,
            },
        },
        "required": ["name", "code", "duration_frames"],
    }

    def __init__(self, remotion_dir: Path):
        self._dir = remotion_dir

    async def execute(
        self,
        name: str,
        code: str,
        duration_frames: int,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
    ) -> str:
        # Validate name is PascalCase-ish (letters/numbers, starts with uppercase)
        if not re.match(r"^[A-Z][A-Za-z0-9]+$", name):
            return f"Error: name must be PascalCase (got '{name}')"

        src_dir = self._dir / "src"
        comp_file = src_dir / f"{name}.tsx"
        root_file = src_dir / "Root.tsx"

        # Write the composition file
        comp_file.write_text(code, encoding="utf-8")

        # Update Root.tsx — add import and Composition entry
        root_content = root_file.read_text(encoding="utf-8")

        # Check if already registered
        if f'id="{name}"' in root_content:
            return f"Composition '{name}' already registered in Root.tsx. Updated {comp_file.name}."

        # Add import after last existing import
        import_line = f'import {{ {name} }} from "./{name}";'
        # Find the last import line
        import_lines = [
            (m.start(), m.end())
            for m in re.finditer(r'^import .+;$', root_content, re.MULTILINE)
        ]
        if import_lines:
            last_import_end = import_lines[-1][1]
            root_content = (
                root_content[:last_import_end]
                + "\n"
                + import_line
                + root_content[last_import_end:]
            )
        else:
            root_content = import_line + "\n" + root_content

        # Add Composition entry before closing </>
        comp_entry = (
            f'      <Composition\n'
            f'        id="{name}"\n'
            f'        component={{{name}}}\n'
            f'        durationInFrames={{{duration_frames}}}\n'
            f'        fps={{{fps}}}\n'
            f'        width={{{width}}}\n'
            f'        height={{{height}}}\n'
            f'      />'
        )
        root_content = root_content.replace("    </>", comp_entry + "\n    </>")

        root_file.write_text(root_content, encoding="utf-8")

        return (
            f"Created {comp_file.name} and registered composition '{name}' in Root.tsx "
            f"({duration_frames} frames, {fps}fps, {width}x{height}). "
            f"Use remotion_render to render it."
        )


class RemotionRenderTool(Tool):
    """Render a Remotion composition to mp4 or gif."""

    name = "remotion_render"
    description = (
        "Render a Remotion composition to mp4 or gif. "
        "The composition must already be registered in Root.tsx "
        "(use remotion_compose first if needed). "
        "Returns a URL to the rendered video."
    )
    parameters = {
        "type": "object",
        "properties": {
            "composition": {
                "type": "string",
                "description": "The composition ID to render (must match an id in Root.tsx)",
            },
            "output_name": {
                "type": "string",
                "description": "Output filename without extension (e.g. 'my-video')",
            },
            "format": {
                "type": "string",
                "description": "Output format: 'mp4' (default) or 'gif'",
                "enum": ["mp4", "gif"],
            },
        },
        "required": ["composition", "output_name"],
    }

    def __init__(self, remotion_dir: Path, base_url: str = "http://localhost:18790"):
        self._dir = remotion_dir
        self._base_url = base_url.rstrip("/")

    async def execute(
        self,
        composition: str,
        output_name: str,
        format: str = "mp4",
    ) -> str:
        out_dir = self._dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        output_file = f"{output_name}.{format}"
        output_path = out_dir / output_file

        cmd = [
            "npx", "remotion", "render",
            "src/index.ts", composition,
            str(output_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Render timed out after 5 minutes for '{composition}'"

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[-2000:]
            return f"Error rendering '{composition}':\n{err}"

        url = f"{self._base_url}/videos/{output_file}"
        return f"Rendered '{composition}' to {format}.\n\nVideo URL: {url}"
