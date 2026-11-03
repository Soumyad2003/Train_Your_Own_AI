import threading
from typing import Literal
from projection import Projection

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from document_store import DocumentStore
from index_factory import make_index

app = FastAPI(title="VectorDB RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],   # React dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)

lock = threading.Lock()          # the store and indexes aren't thread-safe; one request at a time
state = {"index": "hnsw"}
store = DocumentStore(index=make_index(state["index"]))


# ---------- request models ----------
class DocumentIn(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)

class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(3, ge=1, le=20)

class AskIn(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(3, ge=1, le=10)
    max_distance: float = Field(0.4, gt=0)

class IndexIn(BaseModel):
    type: Literal["brute", "kdtree", "hnsw"]


def call_ollama(fn, *args, **kwargs):
    """Turn Ollama connection problems into a clean 503 instead of a 500 crash."""
    try:
        return fn(*args, **kwargs)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama request failed: {e}")


# ---------- endpoints ----------
@app.get("/status")
def status():
    with lock:
        return {
            "ollama": store.ollama.is_available(),
            "index": state["index"],
            "documents": len(store.list_documents()),
            "chunks": len(store.chunks),
        }


@app.post("/documents", status_code=201)
def add_document(body: DocumentIn):
    with lock:
        try:
            doc_id = call_ollama(store.insert_document, body.title, body.text)
        except ValueError as e:                     # e.g. whitespace-only text
            raise HTTPException(status_code=400, detail=str(e))
        chunks = sum(1 for c in store.chunks.values() if c.doc_id == doc_id)
        return {"doc_id": doc_id, "title": body.title, "chunks": chunks}


@app.get("/documents")
def list_documents():
    with lock:
        return store.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    with lock:
        removed = store.delete_document(doc_id)
        if removed == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"doc_id": doc_id, "chunks_removed": removed}


@app.post("/search")
def search(body: SearchIn):
    with lock:
        hits = call_ollama(store.search, body.query, body.k)
        return [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "title": c.title,
                "chunk": c.position,
                "text": c.text,
                "distance": round(d, 4),
            }
            for c, d in hits
        ]


@app.post("/ask")
def ask(body: AskIn):
    with lock:
        return call_ollama(store.ask, body.question, body.k, body.max_distance)


@app.put("/index")
def switch_index(body: IndexIn):
    with lock:
        store.set_index(make_index(body.type))
        state["index"] = body.type
        return {"index": body.type, "vectors": len(store.embeddings)}

class ProjectIn(BaseModel):
    query: str = Field(min_length=1)


_proj = {"key": None, "obj": None}


def get_projection():
    """Recompute the PCA only when the set of stored chunks has changed. Call with the lock held."""
    key = tuple(store.embeddings.keys())
    if _proj["key"] != key:
        ids = list(store.embeddings.keys())
        _proj["obj"] = Projection(ids, [store.embeddings[i] for i in ids]) if ids else None
        _proj["key"] = key
    return _proj["obj"]


@app.get("/map")
def get_map():
    with lock:
        proj = get_projection()
        if proj is None:
            return {"points": [], "explained": [0.0, 0.0]}
        points = []
        for cid, (x, y) in zip(proj.ids, proj.coords):
            c = store.chunks[cid]
            points.append({
                "id": cid, "doc_id": c.doc_id, "title": c.title, "chunk": c.position,
                "x": float(x), "y": float(y),
            })
        return {"points": points, "explained": proj.explained}


@app.post("/map/project")
def project_query(body: ProjectIn):
    with lock:
        proj = get_projection()
        if proj is None:
            return {"x": None, "y": None}
        vec = call_ollama(store.ollama.embed, body.query)
        x, y = proj.project(vec)
        return {"x": x, "y": y}