"""Docling-backed document conversion for the documents workspace."""

from __future__ import annotations

import base64
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_INPUT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".xhtml",
    ".md",
    ".markdown",
    ".adoc",
    ".asciidoc",
    ".tex",
    ".latex",
    ".csv",
    ".txt",
    ".text",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".mp4",
    ".avi",
    ".mov",
    ".vtt",
    ".webvtt",
    ".xml",
    ".xbrl",
    ".json",
}

DEFAULT_EXPORTS = (
    "markdown",
    "text",
    "html",
    "json",
    "doctags",
    "vtt",
    "tables",
    "figures",
    "multimodal",
)


class DoclingUnavailableError(RuntimeError):
    """Raised when the optional Docling runtime is not installed."""


class UnsupportedDoclingFormatError(ValueError):
    """Raised when an uploaded extension is outside Docling's supported set."""


@dataclass(frozen=True)
class DoclingConversionOptions:
    exports: tuple[str, ...] = DEFAULT_EXPORTS
    generate_assets: bool = True
    asset_url_prefix: str = ""
    enable_ocr: bool = True
    force_full_page_ocr: bool = False
    ocr_lang: tuple[str, ...] = ()
    enable_enrichments: bool = False
    enrichments: tuple[str, ...] = ()
    enable_audio: bool = True
    image_scale: float = 2.0
    max_num_pages: int | None = None
    max_file_size: int | None = None


def docling_feature_manifest() -> dict[str, Any]:
    """Return the Docling capabilities the app routes expose."""

    return {
        "converter": "docling",
        "supported_input_extensions": sorted(SUPPORTED_INPUT_EXTENSIONS),
        "supported_output_exports": list(DEFAULT_EXPORTS),
        "conversion_features": [
            "multi-format parsing",
            "layout-aware PDF conversion",
            "OCR for scanned PDFs and images",
            "table structure extraction",
            "figure and page image export",
            "lossless Docling JSON export",
            "Markdown, HTML, plain text, DocTags, and WebVTT export",
            "multimodal page records",
            "audio/video transcription when docling[asr] and ffmpeg are installed",
            "optional code, formula, picture class, and picture description enrichment",
        ],
        "asset_outputs": ["page-images", "figures", "tables-csv", "tables-html"],
        "safe_default": (
            "Imports create editable Tiptap documents from Docling Markdown while "
            "keeping richer conversion metadata and assets available from the response."
        ),
    }


def normalize_docling_exports(raw_exports: str | None) -> tuple[str, ...]:
    if not raw_exports:
        return DEFAULT_EXPORTS
    values = tuple(
        value.strip().lower()
        for value in raw_exports.split(",")
        if value.strip()
    )
    return values or DEFAULT_EXPORTS


def normalize_docling_enrichments(raw_enrichments: str | None) -> tuple[str, ...]:
    if not raw_enrichments:
        return ()
    if raw_enrichments.strip().lower() == "all":
        return ("code", "formula", "picture-classification", "picture-description")
    return tuple(
        value.strip().lower()
        for value in raw_enrichments.split(",")
        if value.strip()
    )


def _ensure_supported(path: Path) -> None:
    name = path.name.lower()
    if name.endswith(".docling.json"):
        return
    if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise UnsupportedDoclingFormatError(f"Unsupported Docling input type: {path.suffix or path.name}")


def _import_docling_converter() -> dict[str, Any]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
            TesseractCliOcrOptions,
        )
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
    except Exception as exc:  # pragma: no cover - exercised through service fallback tests.
        raise DoclingUnavailableError(
            "Docling is not installed in this runtime yet. Install docling or rebuild the "
            "nanobot-api image after accepting the Docling dependency change."
        ) from exc

    return {
        "DocumentConverter": DocumentConverter,
        "ImageFormatOption": ImageFormatOption,
        "InputFormat": InputFormat,
        "PdfFormatOption": PdfFormatOption,
        "PdfPipelineOptions": PdfPipelineOptions,
        "TableStructureOptions": TableStructureOptions,
        "TesseractCliOcrOptions": TesseractCliOcrOptions,
    }


def _set_option(target: Any, name: str, value: Any) -> None:
    if not hasattr(target, name):
        return
    try:
        setattr(target, name, value)
    except Exception:
        return


def _make_pdf_pipeline_options(modules: dict[str, Any], options: DoclingConversionOptions) -> Any:
    pipeline_options = modules["PdfPipelineOptions"]()
    _set_option(pipeline_options, "do_ocr", options.enable_ocr)
    _set_option(pipeline_options, "do_table_structure", True)
    _set_option(pipeline_options, "images_scale", options.image_scale)

    should_keep_images = options.generate_assets or "figures" in options.exports or "multimodal" in options.exports
    _set_option(pipeline_options, "generate_page_images", should_keep_images)
    _set_option(pipeline_options, "generate_picture_images", should_keep_images)

    table_options = modules["TableStructureOptions"](do_cell_matching=True)
    _set_option(pipeline_options, "table_structure_options", table_options)

    if options.force_full_page_ocr or options.ocr_lang:
        ocr_kwargs: dict[str, Any] = {}
        if options.force_full_page_ocr:
            ocr_kwargs["force_full_page_ocr"] = True
        if options.ocr_lang:
            ocr_kwargs["lang"] = list(options.ocr_lang)
        try:
            _set_option(pipeline_options, "ocr_options", modules["TesseractCliOcrOptions"](**ocr_kwargs))
        except Exception:
            pass
        _set_option(pipeline_options, "force_full_page_ocr", options.force_full_page_ocr)

    enrichments = set(options.enrichments)
    if options.enable_enrichments or "code" in enrichments:
        _set_option(pipeline_options, "do_code_enrichment", True)
    if options.enable_enrichments or "formula" in enrichments:
        _set_option(pipeline_options, "do_formula_enrichment", True)
    if options.enable_enrichments or "picture-classification" in enrichments:
        _set_option(pipeline_options, "do_picture_classification", True)
    if options.enable_enrichments or "picture-description" in enrichments:
        _set_option(pipeline_options, "do_picture_description", True)

    return pipeline_options


def _make_audio_format_option(options: DoclingConversionOptions) -> Any | None:
    if not options.enable_audio:
        return None
    try:
        from docling.datamodel import asr_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AsrPipelineOptions
        from docling.document_converter import AudioFormatOption
        from docling.pipeline.asr_pipeline import AsrPipeline
    except Exception:
        return None

    pipeline_options = AsrPipelineOptions()
    if hasattr(asr_model_specs, "WHISPER_TURBO"):
        _set_option(pipeline_options, "asr_options", asr_model_specs.WHISPER_TURBO)
    return InputFormat, AudioFormatOption(pipeline_cls=AsrPipeline, pipeline_options=pipeline_options)


def _create_converter(options: DoclingConversionOptions) -> Any:
    modules = _import_docling_converter()
    InputFormat = modules["InputFormat"]
    pipeline_options = _make_pdf_pipeline_options(modules, options)
    format_options: dict[Any, Any] = {
        InputFormat.PDF: modules["PdfFormatOption"](pipeline_options=pipeline_options),
    }

    if hasattr(InputFormat, "IMAGE"):
        try:
            format_options[InputFormat.IMAGE] = modules["ImageFormatOption"](pipeline_options=pipeline_options)
        except Exception:
            pass

    audio_option = _make_audio_format_option(options)
    if audio_option is not None:
        audio_input_format, option = audio_option
        if hasattr(audio_input_format, "AUDIO"):
            format_options[audio_input_format.AUDIO] = option

    return modules["DocumentConverter"](format_options=format_options)


def _call_export(document: Any, method_name: str, **kwargs: Any) -> Any:
    method = getattr(document, method_name, None)
    if method is None:
        return None
    try:
        return method(**kwargs)
    except TypeError:
        return method()


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    return str(value)


def _asset_url(prefix: str, path: Path) -> str:
    if not prefix:
        return path.name
    return f"{prefix.rstrip('/')}/{path.name}"


def _save_pil_image(image: Any, path: Path) -> bool:
    try:
        image.save(path, format="PNG")
        return True
    except Exception:
        return False


def _export_docling_assets(
    document: Any,
    output_dir: Path | None,
    options: DoclingConversionOptions,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assets: list[dict[str, Any]] = []
    figure_assets: list[dict[str, Any]] = []
    table_assets: list[dict[str, Any]] = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    pages = getattr(document, "pages", {}) or {}
    for page_key, page in pages.items() if isinstance(pages, dict) else []:
        page_no = getattr(page, "page_no", page_key)
        page_image = getattr(getattr(page, "image", None), "pil_image", None)
        if output_dir is not None and page_image is not None:
            filename = f"page-{page_no}.png"
            filepath = output_dir / filename
            if _save_pil_image(page_image, filepath):
                assets.append({
                    "kind": "page",
                    "page_no": page_no,
                    "filename": filename,
                    "url": _asset_url(options.asset_url_prefix, filepath),
                })

    tables = list(getattr(document, "tables", []) or [])
    for index, table in enumerate(tables, start=1):
        table_record: dict[str, Any] = {"kind": "table", "index": index}
        try:
            markdown = None
            dataframe = table.export_to_dataframe(doc=document)
            if hasattr(dataframe, "to_markdown"):
                try:
                    markdown = dataframe.to_markdown(index=False)
                except Exception:
                    markdown = None
            if markdown:
                table_record["markdown"] = markdown
            if output_dir is not None:
                csv_filename = f"table-{index}.csv"
                csv_path = output_dir / csv_filename
                dataframe.to_csv(csv_path, index=False)
                table_record["csv"] = {
                    "filename": csv_filename,
                    "url": _asset_url(options.asset_url_prefix, csv_path),
                }
        except Exception as exc:
            table_record["dataframe_error"] = str(exc)

        try:
            html = table.export_to_html(doc=document)
            if html:
                table_record["html"] = html
                if output_dir is not None:
                    html_filename = f"table-{index}.html"
                    html_path = output_dir / html_filename
                    html_path.write_text(html, encoding="utf-8")
                    table_record["html_file"] = {
                        "filename": html_filename,
                        "url": _asset_url(options.asset_url_prefix, html_path),
                    }
        except Exception as exc:
            table_record["html_error"] = str(exc)
        table_assets.append(table_record)

    if output_dir is not None and hasattr(document, "iterate_items"):
        picture_index = 0
        table_image_index = 0
        try:
            iterator = document.iterate_items()
        except Exception:
            iterator = []
        for element, _level in iterator:
            type_name = type(element).__name__.lower()
            is_picture = "picture" in type_name
            is_table = "table" in type_name
            if not is_picture and not is_table:
                continue
            image_getter = getattr(element, "get_image", None)
            if image_getter is None:
                continue
            try:
                image = image_getter(document)
            except Exception:
                continue
            if image is None:
                continue
            if is_table:
                table_image_index += 1
                kind = "table-image"
                filename = f"table-image-{table_image_index}.png"
            else:
                picture_index += 1
                kind = "figure"
                filename = f"figure-{picture_index}.png"
            filepath = output_dir / filename
            if _save_pil_image(image, filepath):
                record = {
                    "kind": kind,
                    "index": table_image_index if is_table else picture_index,
                    "filename": filename,
                    "url": _asset_url(options.asset_url_prefix, filepath),
                    "label": _jsonable(getattr(element, "label", None)),
                    "page_no": _jsonable(getattr(getattr(element, "prov", [None])[0], "page_no", None)),
                }
                assets.append(record)
                if kind == "figure":
                    figure_assets.append(record)

    return assets, figure_assets, table_assets


def _export_multimodal_pages(conversion_result: Any, page_assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from docling.utils.export import generate_multimodal_pages
    except Exception:
        return []

    page_asset_by_no = {
        str(asset.get("page_no")): asset.get("url")
        for asset in page_assets
        if asset.get("kind") == "page" and asset.get("page_no") is not None
    }
    rows: list[dict[str, Any]] = []
    try:
        iterator = generate_multimodal_pages(conversion_result)
    except Exception:
        return rows

    for content_text, content_md, content_dt, page_cells, page_segments, page in iterator:
        page_no = getattr(page, "page_no", None)
        rows.append({
            "page_no": page_no,
            "page_image_url": page_asset_by_no.get(str(page_no)),
            "contents": content_text,
            "contents_md": content_md,
            "contents_dt": content_dt,
            "cells": _jsonable(page_cells),
            "segments": _jsonable(page_segments),
            "width": _jsonable(getattr(getattr(page, "image", None), "width", None)),
            "height": _jsonable(getattr(getattr(page, "image", None), "height", None)),
        })
    return rows


def _metadata_from_result(conversion_result: Any, document: Any, elapsed_seconds: float) -> dict[str, Any]:
    input_info = getattr(conversion_result, "input", None)
    return {
        "status": str(getattr(conversion_result, "status", "")),
        "version": _jsonable(getattr(conversion_result, "version", None)),
        "document_hash": _jsonable(getattr(input_info, "document_hash", None)),
        "page_count": len(getattr(document, "pages", {}) or {}),
        "text_count": len(getattr(document, "texts", []) or []),
        "table_count": len(getattr(document, "tables", []) or []),
        "picture_count": len(getattr(document, "pictures", []) or []),
        "key_value_count": len(getattr(document, "key_value_items", []) or []),
        "timings": _jsonable(getattr(conversion_result, "timings", None)),
        "confidence": _jsonable(getattr(conversion_result, "confidence", None)),
        "errors": _jsonable(getattr(conversion_result, "errors", None)),
        "conversion_seconds": round(elapsed_seconds, 3),
    }


def convert_document_with_docling(
    source_path: Path,
    *,
    original_filename: str | None = None,
    output_dir: Path | None = None,
    options: DoclingConversionOptions | None = None,
) -> dict[str, Any]:
    """Convert a local document path with Docling and return app-ready exports."""

    source_path = Path(source_path)
    _ensure_supported(Path(original_filename or source_path.name))
    options = options or DoclingConversionOptions()
    conversion_id = output_dir.name if output_dir is not None else uuid.uuid4().hex
    started = time.perf_counter()
    converter = _create_converter(options)

    convert_kwargs: dict[str, Any] = {"raises_on_error": False}
    if options.max_num_pages:
        convert_kwargs["max_num_pages"] = options.max_num_pages
    if options.max_file_size:
        convert_kwargs["max_file_size"] = options.max_file_size

    conversion_result = converter.convert(source_path, **convert_kwargs)
    document = getattr(conversion_result, "document", None)
    if document is None:
        raise RuntimeError("Docling did not return a document for this input.")

    requested = set(options.exports or DEFAULT_EXPORTS)
    exports: dict[str, Any] = {}
    if "markdown" in requested:
        exports["markdown"] = _call_export(document, "export_to_markdown", traverse_pictures=True)
    if "text" in requested:
        exports["text"] = _call_export(document, "export_to_text", traverse_pictures=True)
    if "html" in requested:
        exports["html"] = _call_export(document, "export_to_html")
    if "json" in requested:
        exports["json"] = _jsonable(_call_export(document, "export_to_dict"))
    if "doctags" in requested:
        exports["doctags"] = _call_export(document, "export_to_doctags")
    if "vtt" in requested:
        vtt = _call_export(document, "export_to_vtt")
        if vtt:
            exports["vtt"] = vtt

    assets, figure_assets, table_assets = _export_docling_assets(
        document,
        output_dir if options.generate_assets else None,
        options,
    )
    if "tables" in requested:
        exports["tables"] = table_assets
    if "figures" in requested:
        exports["figures"] = figure_assets
    if "multimodal" in requested:
        exports["multimodal_pages"] = _export_multimodal_pages(conversion_result, assets)

    elapsed = time.perf_counter() - started
    metadata = _metadata_from_result(conversion_result, document, elapsed)
    filename = original_filename or source_path.name
    document_title = str(getattr(document, "name", "") or "").strip()
    filename_title = Path(filename).stem.strip()
    if document_title and document_title.lower() not in {"upload", source_path.stem.lower()}:
        title = document_title
    else:
        title = filename_title or document_title or "Imported document"

    result = {
        "success": True,
        "converter": "docling",
        "conversion_id": conversion_id,
        "filename": filename,
        "title": title,
        "exports": exports,
        "assets": assets,
        "metadata": metadata,
        "markdown": exports.get("markdown"),
        "text": exports.get("text"),
        "html": exports.get("html"),
        "document": exports.get("json"),
        "doctags": exports.get("doctags"),
        "tables": table_assets,
        "figures": figure_assets,
        "multimodal_pages": exports.get("multimodal_pages", []),
    }
    return _jsonable(result)
