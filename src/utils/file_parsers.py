"""Parse non-tabular uploads (PDF, etc.) for context. Tabular (CSV, JSON, XLSX) are handled by data_loader."""

import os
from typing import Optional


def extract_pdf_text(file_path: str, max_chars: int = 50000) -> Optional[str]:
    """Extract text from a PDF for use as context. Returns None on failure or if empty."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        parts = []
        n = 0
        for page in reader.pages:
            if n >= 100:  # cap pages
                break
            text = page.extract_text()
            if text:
                parts.append(text)
            n += 1
        out = "\n".join(parts).strip()
        if not out:
            return None
        if len(out) > max_chars:
            out = out[:max_chars] + "\n[... truncated ...]"
        return out
    except Exception as e:
        print(f"[file_parsers] PDF extract failed for {file_path}: {e}")
        return None


def get_upload_context_for_non_tabular(
    file_paths: list[str],
) -> str:
    """
    Build a single context string from non-tabular uploads (PDF text, image placeholders).
    Used to answer the user's question with respect to uploaded files.
    """
    if not file_paths:
        return ""
    lines = ["## User uploaded the following file(s) (use this content to answer the question):"]
    for path in file_paths:
        if not path or not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        lower = name.lower()
        if lower.endswith(".pdf"):
            text = extract_pdf_text(path)
            if text:
                lines.append(f"\n### PDF: {name}\n```\n{text}\n```")
            else:
                lines.append(f"\n### PDF: {name} (could not extract text)")
        elif lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            lines.append(f"\n### Image: {name} (user uploaded an image; describe or answer based on filename/context if you cannot analyze the image)")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)
