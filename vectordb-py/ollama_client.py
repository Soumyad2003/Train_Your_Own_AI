import requests

class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434,
                 embed_model="nomic-embed-text", gen_model="llama3.2"):
        self.base = f"http://{host}:{port}"
        self.embed_model = embed_model
        self.gen_model = gen_model
        self._cache: dict[str, list[float]] = {}

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.base}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for one string (768 floats for nomic-embed-text)."""
        if text in self._cache:
            return self._cache[text]
        r = requests.post(
            f"{self.base}/api/embed",
            json={"model": self.embed_model, "input": text},
            timeout=60,
        )
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        if len(self._cache) >= 512:                      # keep it bounded
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec

    def generate(self, prompt: str) -> str:
        """Return the full LLM answer as one string (no streaming yet)."""
        r = requests.post(
            f"{self.base}/api/generate",
            json={"model": self.gen_model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"]