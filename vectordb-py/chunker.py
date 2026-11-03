def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    """Split text into chunks of ~chunk_size words, each sharing `overlap` words with the previous one."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):   # this chunk reached the end, stop
            break
    return chunks