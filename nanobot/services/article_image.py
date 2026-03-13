"""Generate Canva-style article images (cover + body) using Pillow.

Replicates the Street Voices 1080×1350 article template (2-page):
  Page 1 — full-bleed hero, heavy dark gradient, yellow banner, headline, CTA pill
  Page 2 — white background, centred body copy, bold subheadings, CTA pill
"""

from __future__ import annotations

import re
import uuid
from io import BytesIO
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

# --- Constants -----------------------------------------------------------

WIDTH, HEIGHT = 1080, 1350

YELLOW = (247, 201, 72)       # #F7C948
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

_FONTS_DIR = Path.home() / ".nanobot" / "workspace" / "assets" / "fonts"
_OUTPUT_DIR = Path.home() / ".nanobot" / "workspace" / "article-images"

# Archivo Black is wider/blockier — closest free match to Cocogoose Classic
_HEADLINE_FONT = _FONTS_DIR / "ArchivoBlack-Regular.ttf"
_HEADLINE_FALLBACK = _FONTS_DIR / "Anton-Regular.ttf"
_RUBIK = _FONTS_DIR / "Rubik-Regular.ttf"
_RUBIK_BOLD = _FONTS_DIR / "Rubik-Bold.ttf"


def _load_font(path: Path, size: int, fallback: Path | None = None) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        if fallback:
            try:
                return ImageFont.truetype(str(fallback), size)
            except OSError:
                pass
        logger.warning("Font {} not found, using default", path)
        return ImageFont.load_default()


# --- Helpers -------------------------------------------------------------


def _cover_resize(img: Image.Image) -> Image.Image:
    """Crop-to-fill: scale so shortest side matches, then centre-crop."""
    src_w, src_h = img.size
    scale = max(WIDTH / src_w, HEIGHT / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - WIDTH) // 2
    top = (new_h - HEIGHT) // 2
    return img.crop((left, top, left + WIDTH, top + HEIGHT))


def _make_gradient(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """Heavy bottom gradient — nearly opaque black in the lower 40%.

    Matches the Canva template: transparent up top, begins fading around 30%,
    darkens aggressively, and is near-solid black from ~70% to the bottom.
    """
    start_y = int(height * 0.30)
    for y in range(start_y, height):
        progress = (y - start_y) / (height - start_y)
        # Two-phase curve: gentle start, then very steep
        if progress < 0.35:
            alpha = int(60 * (progress / 0.35))
        elif progress < 0.6:
            t = (progress - 0.35) / 0.25
            alpha = int(60 + 140 * t)
        else:
            # Bottom 40%: near-solid black
            t = (progress - 0.6) / 0.4
            alpha = int(200 + 55 * t)
        alpha = min(alpha, 255)
        draw.rectangle([(0, y), (width, y + 1)], fill=(0, 0, 0, alpha))


def _draw_banner(img: Image.Image, text: str = "ARTICLE") -> None:
    """Yellow vertical pill banner at top-left with rotated white text.

    Canva specs: Rubik 20.6pt, text at X=148.3 Y=214.2, W=142.6 H=32.4,
    rotated 90°. The banner is a rounded pill shape, ~56×272px.
    """
    banner_w, banner_h = 56, 272
    banner = Image.new("RGBA", (banner_w, banner_h), YELLOW)

    font = _load_font(_RUBIK_BOLD, 21)  # Canva 20.6pt → round to 21
    # Create text on a wide canvas, then rotate 90° CCW
    txt_img = Image.new("RGBA", (banner_h, banner_w), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    bbox = txt_draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Post-render nudge: align label with pre-render HTML reference
    # (text was visually too close to the right edge of the pill).
    text_x = (banner_h - tw) // 2
    text_y = ((banner_w - th) // 2) - 15
    txt_draw.text((text_x, text_y), text, fill=WHITE, font=font)
    rotated = txt_img.rotate(90, expand=True)
    rx, ry = rotated.size
    banner.paste(rotated, ((banner_w - rx) // 2, (banner_h - ry) // 2), rotated)
    # Canva: banner bleeds off the left edge — ~20px off-canvas
    img.paste(banner, (-20, 0), banner)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """Word-wrap text to fit within max_width."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_pill_cta(
    img: Image.Image,
    y_center: int,
    text: str = "Go visit streetvoices.ca to read more",
    font_size: int = 22,
) -> None:
    """Yellow pill-shaped CTA button, centred horizontally at y_center."""
    font = _load_font(_RUBIK, font_size)
    tmp_draw = ImageDraw.Draw(img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 48, 18
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2
    x = (WIDTH - pill_w) // 2
    y = y_center - pill_h // 2

    pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    pill_draw = ImageDraw.Draw(pill)
    pill_draw.rounded_rectangle(
        [(0, 0), (pill_w - 1, pill_h - 1)],
        radius=pill_h // 2,
        fill=YELLOW,
    )
    # Centre text vertically in pill
    pill_draw.text((pad_x, pad_y - 2), text, fill=BLACK, font=font)
    img.paste(pill, (x, y), pill)


# --- Page renderers ------------------------------------------------------


def _render_cover(
    hero: Image.Image, headline: str, category: str = "ARTICLE",
) -> Image.Image:
    """Page 1: full-bleed hero, heavy gradient, banner, headline, CTA."""
    base = _cover_resize(hero.convert("RGBA"))

    # Heavy dark gradient overlay
    gradient = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    _make_gradient(ImageDraw.Draw(gradient), WIDTH, HEIGHT)
    base = Image.alpha_composite(base, gradient)

    # Yellow vertical banner
    _draw_banner(base, text=category.upper())

    # Headline — Canva uses Cocogoose Compressed 51pt at X=98, Y=875, W=874
    # Archivo Black is closest free match; size tuned to fill same W=874 box
    draw = ImageDraw.Draw(base)
    font = _load_font(_HEADLINE_FONT, 51, fallback=_HEADLINE_FALLBACK)
    margin_left = 98   # Canva X=98
    max_w = 874        # Canva text-box width
    lines = _wrap_text(draw, headline.upper(), font, max_w)

    # Canva line spacing = 1.0 → line_height ≈ font size * 1.3 (default leading)
    line_height = 64
    # Canva CTA text at Y=1190.6 → pill centre ≈ 1206
    cta_y = 1206
    headline_bottom = cta_y - 46
    total_text_h = line_height * len(lines)
    y_start = headline_bottom - total_text_h

    for i, line in enumerate(lines):
        draw.text(
            (margin_left, y_start + i * line_height),
            line, fill=WHITE, font=font,
        )

    # CTA pill — Canva: Rubik 20.2pt, centred, Y≈1206
    _draw_pill_cta(base, y_center=cta_y, font_size=20)

    return base.convert("RGB")


def _draw_mixed_line(
    draw: ImageDraw.ImageDraw,
    y: int,
    bold_part: str,
    regular_part: str,
    font_bold: ImageFont.FreeTypeFont,
    font_regular: ImageFont.FreeTypeFont,
    max_w: int,
) -> int:
    """Draw a line with a bold prefix and regular suffix, centred."""
    bb = draw.textbbox((0, 0), bold_part, font=font_bold)
    bold_w = bb[2] - bb[0]
    br = draw.textbbox((0, 0), regular_part, font=font_regular)
    reg_w = br[2] - br[0]
    total_w = bold_w + reg_w
    x = (WIDTH - total_w) // 2
    draw.text((x, y), bold_part, fill=BLACK, font=font_bold)
    draw.text((x + bold_w, y), regular_part, fill=BLACK, font=font_regular)
    return total_w


# Pattern to detect interview-style labels like "Street Voices:" or "Derin Falana:"
_LABEL_RE = re.compile(r'^([A-Z][A-Za-z\s]+:)\s*(.+)$')


def _render_body(body_text: str) -> Image.Image:
    """Page 2: white background, centred body text, bold subheadings, CTA.

    Canva specs: Rubik 23.6pt, X=159.5 Y=174.3, W=760.9 H=925,
    Center-align, Letter spacing: 0, Line spacing: 1.5
    """
    base = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(base)

    font_regular = _load_font(_RUBIK, 24)   # Canva 23.6pt → round to 24
    font_bold = _load_font(_RUBIK_BOLD, 24)

    margin_left = 160    # Canva X=159.5 → 160
    max_w = 761          # Canva W=760.9 → 761
    y = 174              # Canva Y=174.3 → 174
    line_height = 36     # Canva 23.6pt * 1.5 line spacing ≈ 35.4 → 36

    paragraphs = body_text.split("\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            y += int(line_height * 0.4)  # paragraph gap
            continue

        # Detect subheadings: short lines in title case or all caps
        is_heading = len(para) < 60 and (para.isupper() or para.istitle())

        if is_heading:
            y += 8  # extra space before heading
            wrapped = _wrap_text(draw, para, font_bold, max_w)
            for line in wrapped:
                if y + line_height > 1099:
                    break
                bbox = draw.textbbox((0, 0), line, font=font_bold)
                lw = bbox[2] - bbox[0]
                x = (WIDTH - lw) // 2
                draw.text((x, y), line, fill=BLACK, font=font_bold)
                y += line_height
            continue

        # Check for interview-style label (e.g. "Street Voices: ..." or "Derin Falana: ...")
        label_match = _LABEL_RE.match(para)
        if label_match:
            label = label_match.group(1)
            rest = label_match.group(2)

            # Measure bold label width
            lb = draw.textbbox((0, 0), label + " ", font=font_bold)
            label_w = lb[2] - lb[0]

            # Wrap remaining text to fit after the label on the first line
            first_max = max_w - label_w
            rest_words = rest.split()
            first_line_rest = ""
            remaining_words = list(rest_words)
            for wi, word in enumerate(rest_words):
                test = f"{first_line_rest} {word}".strip()
                tb = draw.textbbox((0, 0), test, font=font_regular)
                if tb[2] - tb[0] <= first_max:
                    first_line_rest = test
                    remaining_words = rest_words[wi + 1:]
                else:
                    remaining_words = rest_words[wi:]
                    break

            # Draw first line: bold label + regular text
            total_first_w = label_w + draw.textbbox((0, 0), first_line_rest, font=font_regular)[2]
            x_start = (WIDTH - total_first_w) // 2
            if y + line_height <= 1099:
                draw.text((x_start, y), label + " ", fill=BLACK, font=font_bold)
                draw.text((x_start + label_w, y), first_line_rest, fill=BLACK, font=font_regular)
                y += line_height

            # Wrap and draw remaining text
            if remaining_words:
                leftover = " ".join(remaining_words)
                wrapped = _wrap_text(draw, leftover, font_regular, max_w)
                for line in wrapped:
                    if y + line_height > 1099:
                        break
                    bbox = draw.textbbox((0, 0), line, font=font_regular)
                    lw = bbox[2] - bbox[0]
                    x = (WIDTH - lw) // 2
                    draw.text((x, y), line, fill=BLACK, font=font_regular)
                    y += line_height
            continue

        # Regular paragraph
        wrapped = _wrap_text(draw, para, font_regular, max_w)
        for line in wrapped:
            if y + line_height > 1099:
                break
            bbox = draw.textbbox((0, 0), line, font=font_regular)
            lw = bbox[2] - bbox[0]
            x = (WIDTH - lw) // 2
            draw.text((x, y), line, fill=BLACK, font=font_regular)
            y += line_height

        if y + line_height > 1099:  # Canva text box ends at Y=174+925=1099
            break

    # CTA pill at bottom — same style as cover page
    rgba = base.convert("RGBA")
    _draw_pill_cta(rgba, y_center=1206, font_size=20)
    return rgba.convert("RGB")


# --- Public API ----------------------------------------------------------


async def generate_article_images(
    headline: str,
    body_text: str,
    hero_image_url: str,
    category: str = "ARTICLE",
) -> dict:
    """Generate cover + body PNGs. Returns image_id and file paths."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex[:12]

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(hero_image_url)
        resp.raise_for_status()
        hero = Image.open(BytesIO(resp.content))

    cover = _render_cover(hero, headline, category)
    body = _render_body(body_text)

    cover_path = _OUTPUT_DIR / f"{image_id}_cover.png"
    body_path = _OUTPUT_DIR / f"{image_id}_body.png"
    cover.save(str(cover_path), "PNG", optimize=True)
    body.save(str(body_path), "PNG", optimize=True)

    logger.info("Generated article images: {}", image_id)
    return {
        "image_id": image_id,
        "cover_url": f"/article-images/{image_id}_cover.png",
        "body_url": f"/article-images/{image_id}_body.png",
    }
