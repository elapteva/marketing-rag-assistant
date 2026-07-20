import re

def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if not text.strip():
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk settings.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            candidate = text[start:end]
            boundary = max(candidate.rfind(". "), candidate.rfind("\n"), candidate.rfind(" "))
            if boundary > chunk_size * 0.6:
                end = start + boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

