from __future__ import annotations

import urllib.request
from pathlib import Path


def download_pdf(url: str, destination: Path, timeout: int = 60) -> Path:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-claw/0.1 (+https://github.com/zejun04/paper_claw)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        destination.write_bytes(response.read())
    return destination


def extract_text(path: Path, max_characters: int = 90000) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF，无法读取 PDF") from exc

    document = fitz.open(path)
    try:
        text = "\n\n".join(page.get_text("text") for page in document)
    finally:
        document.close()
    return text[:max_characters]
