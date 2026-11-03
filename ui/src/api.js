const BASE = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error("Can't reach the API. Is uvicorn running?");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : JSON.stringify(data?.detail ?? res.statusText);
    throw new Error(detail);
  }
  return data;
}

const post = (path, body) =>
  request(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  benchmark: (n, dim, ef) => post("/benchmark", { n, dim, ef }),
  map: () => request("/map"),
  project: (query) => post("/map/project", { query }),
  status: () => request("/status"),
  listDocuments: () => request("/documents"),
  addDocument: (title, text) => post("/documents", { title, text }),
  deleteDocument: (id) => request(`/documents/${id}`, { method: "DELETE" }),
  search: (query, k = 3) => post("/search", { query, k }),
  ask: (question, k = 3, max_distance = 0.4) =>
    post("/ask", { question, k, max_distance }),
  setIndex: (type) =>
    request("/index", { method: "PUT", body: JSON.stringify({ type }) }),
};
