from dataclasses import dataclass
from chunker import chunk_text
from ollama_client import OllamaClient
from index_brute import BruteForceIndex


@dataclass
class Chunk:
    id: int
    doc_id: int
    title: str
    text: str
    position: int          # chunk's order within its document


class DocumentStore:
    def __init__(self, ollama=None, index=None):
        self.ollama = ollama or OllamaClient()
        self.index = index or BruteForceIndex("cosine")   # swappable later (KD-Tree / HNSW)
        self.chunks: dict[int, Chunk] = {}
        self.embeddings: dict[int, list[float]] = {}
        self._next_chunk_id = 0
        self._next_doc_id = 0

    def insert_document(self, title: str, text: str,
                        chunk_size: int = 250, overlap: int = 50) -> int:
        pieces = chunk_text(text, chunk_size, overlap)
        if not pieces:
            raise ValueError("Document is empty")

        # Embed everything first, so an Ollama failure can't leave a half-inserted document.
        embeddings = [self.ollama.embed(p) for p in pieces]

        doc_id = self._next_doc_id
        self._next_doc_id += 1
        for position, (piece, emb) in enumerate(zip(pieces, embeddings)):
            cid = self._next_chunk_id
            self._next_chunk_id += 1
            self.chunks[cid] = Chunk(cid, doc_id, title, piece, position)
            self.embeddings[cid] = emb
            self.index.insert(cid, emb)
        return doc_id

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        qvec = self.ollama.embed(query)
        return [(self.chunks[cid], dist) for cid, dist in self.index.search(qvec, k)]

    PROMPT_TEMPLATE = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know. "
        "Be concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def ask(self, question: str, k: int = 3, max_distance: float = 0.4) -> dict:
        hits = [(c, d) for c, d in self.search(question, k) if d <= max_distance]

        if not hits:
            return {
                "answer": "I couldn't find anything relevant in your documents.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[{i}] ({c.title})\n{c.text}" for i, (c, _) in enumerate(hits, start=1)
        )
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)

        return {
            "answer": self.ollama.generate(prompt).strip(),
            "sources": [
                {"title": c.title, "chunk": c.position, "distance": round(d, 3)}
                for c, d in hits
            ],
        }

    def delete_document(self, doc_id: int) -> int:
        """Remove all chunks of a document. Returns how many chunks were removed."""
        ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        for cid in ids:
            self.index.delete(cid)
            del self.chunks[cid]
            self.embeddings.pop(cid, None)
        return len(ids)

    def list_documents(self) -> list[dict]:
        docs: dict[int, dict] = {}
        for c in self.chunks.values():
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())

    def set_index(self, index) -> None:
        """Swap the search index, re-inserting every stored embedding."""
        for cid, emb in self.embeddings.items():
            index.insert(cid, emb)      # fully built before we swap, so a failure keeps the old index
        self.index = index