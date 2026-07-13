"""
pdf_to_chunks.py — ADTC 2026 Agriculture RAG: corpus extraction + chunking

WHAT THIS DOES (plain English version, for explaining to your brother):
  You point it at a folder of agriculture PDFs. It reads each one, pulls out
  the raw text, and cuts that text into bite-sized "chunks" — small enough
  that the model can actually read one at inference time, but big enough
  that each chunk still makes sense on its own (a whole section about maize
  fertilizer rates, not half a sentence). It saves those chunks as JSON so
  the next script (embedding + FAISS index) can turn them into a searchable
  vector store.

WHY CHUNK ON HEADINGS INSTEAD OF FIXED TOKEN WINDOWS:
  Fixed-size windows (e.g. "every 500 characters") cut sentences and even
  crop-specific sections in half — a chunk might end mid-way through the
  tomato fertilizer rate and start the next chunk mid-way through pest
  control for a different crop. Since these ARC/NDA guides already have
  clean section headers per crop, we chunk on those boundaries first, and
  only fall back to size-based splitting if a section is too long.

USAGE:
  python pdf_to_chunks.py --input corpus/raw --output corpus/chunks.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


# --- Config -----------------------------------------------------------------

# Max characters per chunk. We keep this conservative because the embedding
# server is stricter than the model context window, and we want each chunk to
# stay comfortably below the server's token safety limit on an 8GB laptop.
MAX_CHUNK_CHARS = 1300

# Minimum chunk size — anything shorter than this gets merged into the
# next chunk instead of being retrieved on its own (a 2-line orphan chunk
# with no context is useless to the model).
MIN_CHUNK_CHARS = 250

# Overlap is kept small but meaningful so adjacent chunks share context
# without making the next chunk too large for the embedding endpoint.
OVERLAP_CHARS = 180

# Headings in these government/ARC guides are typically ALL CAPS lines,
# or short lines (under ~60 chars) ending without a period, often followed
# by a colon. This regex is intentionally loose — better to over-detect
# section breaks than under-detect them, since merging is cheap (see
# MIN_CHUNK_CHARS above) but mis-splitting mid-sentence is not.
HEADING_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s/&\-]{4,60}|(?:\d+\.\s?)?[A-Z][a-zA-Z\s/&\-]{4,50}:)\s*$"
)


# --- Extraction ---------------------------------------------------------------

def extract_text_by_page(pdf_path: Path) -> list[str]:
    """
    Pull raw text out of a PDF, one string per page.

    Why per-page and not one giant blob: we tag each chunk with its source
    page number later, so when the model cites "planting dates" you can
    trace it back to page 4 of the maize guide, not just "somewhere in
    this 40-page document."
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Collapse the multiple-spaces/broken-line mess that PDF text
        # extraction often produces from multi-column layouts.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return pages


# --- Chunking -----------------------------------------------------------------

def split_into_sections(page_text: str) -> list[str]:
    """
    Break one page's text into sections using heading-like lines as the
    split points. If no headings are detected, the whole page comes back
    as a single section (handled by the length-based fallback next).
    """
    lines = page_text.split("\n")
    sections = []
    current: list[str] = []

    for line in lines:
        if HEADING_PATTERN.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [s for s in sections if s]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph into chunks without cutting mid-sentence."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) > 1:
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = sentence
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk]

    words = text.split()
    if len(words) <= 1:
        return [text[:max_chars]]

    chunks = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def split_text_with_overlap(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    """
    Split plain text into smaller chunks with a small overlap.

    The splitter prefers paragraph boundaries, falls back to sentence
    boundaries for very long paragraphs, and keeps each chunk comfortably
    under the embedding endpoint's safe size.
    """
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    current = ""
    previous_tail = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                previous_tail = chunks[-1][-overlap_chars:] if overlap_chars else ""
                current = ""
            child_chunks = _split_long_text(paragraph, max_chars)
            if child_chunks:
                chunks.extend(child_chunks)
                previous_tail = child_chunks[-1][-overlap_chars:] if overlap_chars else ""
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            previous_tail = chunks[-1][-overlap_chars:] if overlap_chars else ""

        overlap = previous_tail
        if overlap and len(overlap) + len(paragraph) + 2 > max_chars:
            overlap = overlap[-max(40, max_chars // 4):]

        candidate = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
        if len(candidate) > max_chars:
            split_chunks = _split_long_text(candidate, max_chars)
            if split_chunks:
                chunks.extend(split_chunks[:-1])
                current = split_chunks[-1].strip()
                previous_tail = current[-overlap_chars:] if overlap_chars else ""
                continue
            current = candidate[:max_chars].strip()
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def clean_chunk_text(text: str) -> str:
    """Remove obvious boilerplate and repetitive noise from a chunk before indexing."""
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    lines = [line for line in lines if not re.fullmatch(r"(?:page|p\.)\s*\d+", line, flags=re.I)]
    lines = [line for line in lines if not re.fullmatch(r"(?:department|republic|ministry|government)\s*[:\-].*", line, flags=re.I)]
    lines = [line for line in lines if not re.fullmatch(r"(?:agriculture|south africa|cape town|pretoria|republic of south africa)", line, flags=re.I)]
    lines = [line for line in lines if not re.search(r"\brepublic\b", line, flags=re.I)]

    filtered: list[str] = []
    for line in lines:
        if not filtered or filtered[-1].lower() != line.lower():
            filtered.append(line)

    cleaned = "\n".join(filtered).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned)
    return cleaned.strip()


def enforce_size_limits(sections: list[str]) -> list[str]:
    """
    Two cleanup passes over the heading-based sections:
      1. Anything over MAX_CHUNK_CHARS gets split further, on paragraph
         breaks where possible, so we never blow the embedding budget.
      2. Anything under MIN_CHUNK_CHARS gets merged into the previous chunk
         only when the combined size still fits safely inside the limit.
    """
    cleaned_sections = [clean_chunk_text(section) for section in sections]
    cleaned_sections = [section for section in cleaned_sections if section and len(section) >= 40]

    sized: list[str] = []
    for section in cleaned_sections:
        if len(section) <= MAX_CHUNK_CHARS:
            sized.append(section)
            continue
        sized.extend(split_text_with_overlap(section))

    merged: list[str] = []
    for chunk in sized:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(chunk) < MIN_CHUNK_CHARS:
            combined = f"{merged[-1]}\n\n{chunk}".strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                merged[-1] = combined
                continue
        merged.append(chunk)

    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= MAX_CHUNK_CHARS:
            final.append(chunk)
        else:
            final.extend(split_text_with_overlap(chunk))

    return [chunk for chunk in final if chunk and len(chunk) >= 40]


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """
    Full pipeline for one PDF: extract -> split into headed sections per
    page -> enforce size limits -> attach metadata each chunk needs for
    citation and filtering later (source file, page number, chunk index).
    """
    pages = extract_text_by_page(pdf_path)
    chunks = []
    chunk_idx = 0

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        sections = split_into_sections(page_text)
        sized_sections = enforce_size_limits(sections)
        for section in sized_sections:
            if len(section.strip()) < 30:  # skip near-empty noise
                continue
            chunks.append({
                "id": f"{pdf_path.stem}_{chunk_idx:04d}",
                "source": pdf_path.name,
                "page": page_num,
                "text": section.strip(),
            })
            chunk_idx += 1

    return chunks


# --- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder of raw PDFs")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        chunks = chunk_pdf(pdf_path)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
