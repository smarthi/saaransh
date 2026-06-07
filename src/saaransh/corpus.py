"""Corpus loading.

Sources:
  - load_local(dir): a folder of page images and/or PDFs + queries.jsonl. PDFs are
    rendered to page images; each page is named "<pdfstem>_p<n>" so queries.jsonl can
    reference it. Each line: {"query": "...", "image": "report_p3"} (extension optional).
  - load_vidore(name): a ViDoRe benchmark subset via `datasets` (needs the vidore extra).

Both return (images, queries, qrels) where qrels maps query idx -> {relevant doc idx}.
render_document() is exposed for ad-hoc use (render a PDF/image to page images).
"""

from __future__ import annotations

import io
import json
from pathlib import Path


def pdf_to_images(pdf_path: str | Path, dpi: int = 175):
    """Render every page of a PDF to a PIL RGB image via PyMuPDF."""
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    images = []
    scale = dpi / 72.0  # 72 dpi is the PDF base; scale to target dpi
    mat = fitz.Matrix(scale, scale)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(Image.open(io.BytesIO(pix.tobytes("ppm"))).convert("RGB"))
    doc.close()
    return images


def render_document(path: str | Path, dpi: int = 175):
    """Render a PDF or image file to a list of PIL RGB page images."""
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_to_images(path, dpi=dpi)
    return [Image.open(path).convert("RGB")]


def load_local(root: str | Path, dpi: int = 175):
    root = Path(root)
    rows = [
        json.loads(line)
        for line in (root / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # Build an ordered page list from images + PDFs; name pages so qrels can reference them.
    names: list[str] = []
    images: list = []
    from PIL import Image

    for p in sorted(root.glob("*.png")) + sorted(root.glob("*.jpg")):
        names.append(p.stem)
        images.append(Image.open(p).convert("RGB"))
    for pdf in sorted(root.glob("*.pdf")):
        for i, img in enumerate(pdf_to_images(pdf, dpi=dpi)):
            names.append(f"{pdf.stem}_p{i}")
            images.append(img)

    name_to_idx = {n: i for i, n in enumerate(names)}

    def resolve(ref: str) -> int:
        return name_to_idx[Path(ref).stem]

    queries = [r["query"] for r in rows]
    qrels = {i: {resolve(r["image"])} for i, r in enumerate(rows)}
    return images, queries, qrels


def load_vidore(name: str = "vidore/docvqa_test_subsampled", split: str = "test", limit: int | None = None):
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    images, queries, qrels = [], [], {}
    for qi, row in enumerate(ds):
        images.append(row["image"].convert("RGB"))
        queries.append(row["query"])
        qrels[qi] = {qi}  # aligned 1:1 in the subsampled sets
    return images, queries, qrels
