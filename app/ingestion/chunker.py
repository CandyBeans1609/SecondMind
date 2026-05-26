from config import settings


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split a list of page dicts into smaller overlapping chunks.

    Input:  [{"text": str, "page": int, "source": str, "content_type": str, "heading": str}, ...]
    Output: [{"text": str, "page": int, "source": str, "chunk_index": int, ...}, ...]
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = _split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk["text"],
                "page": page["page"],
                "source": page["source"],
                "chunk_index": chunk_index,
                "chunk_total": None,  # fill in after loop
                "char_start": chunk["start"],
                "char_end": chunk["end"],
                "content_type": page.get("content_type", "text"),  # "text" | "table"
                "section_heading": page.get("heading", ""),
            })
            chunk_index += 1

    # Fill in chunk_total after loop
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["chunk_total"] = total_chunks

    return chunks


# ── Private ────────────────────────────────────────────────────────────────────

def _split_text(text: str) -> list[dict]:
    """
    Sliding window splitter.
    Tries to break on sentence boundaries ('. ') within the window
    to avoid cutting mid-sentence.
    Returns a list of dicts with {"text": str, "start": int, "end": int}.
    """
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    step = size - overlap

    if len(text) <= size:
        return [{"text": text.strip(), "start": 0, "end": len(text)}]

    chunks = []
    start = 0

    while start < len(text):
        end = start + size

        if end < len(text):
            # Try to snap to the last sentence boundary in the window
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "start": start,
                "end": min(end, len(text)),
            })

        start += step
        if start >= len(text):
            break

    return chunks