import requests

BASE = "http://127.0.0.1:8000"

print("status:", requests.get(f"{BASE}/status").json())

docs = {
    "Data Structures": "A stack is a last-in first-out data structure. You push items onto the top "
                       "and pop them off the top. A queue is first-in first-out. Binary trees "
                       "organise data hierarchically, and hash tables give average constant-time lookups.",
    "Cooking Pasta": "To cook pasta, boil water with plenty of salt, add the pasta, and cook until "
                     "al dente. Drain it and toss it with sauce straight away.",
    "Photosynthesis": "Photosynthesis is how plants convert sunlight, water and carbon dioxide into "
                      "glucose and oxygen. It happens in the chloroplasts, where chlorophyll absorbs light.",
}
ids = {}
for title, text in docs.items():
    r = requests.post(f"{BASE}/documents", json={"title": title, "text": text})
    ids[title] = r.json()["doc_id"]
    print(f"insert {title!r}: {r.status_code}", r.json())

print("\nlist:", requests.get(f"{BASE}/documents").json())

print("\nsearch through each index type:")
for kind in ("brute", "kdtree", "hnsw"):
    requests.put(f"{BASE}/index", json={"type": kind})
    hits = requests.post(f"{BASE}/search", json={"query": "How does a stack work?", "k": 2}).json()
    print(f"  {kind:7s}", [(h["title"], h["distance"]) for h in hits])

print("\nask:")
r = requests.post(f"{BASE}/ask", json={"question": "How do I cook pasta properly?"})
print(" ", r.json())
r = requests.post(f"{BASE}/ask", json={"question": "Who won the 2018 football World Cup?"})
print(" ", r.json())

print("\nerrors:")
print("  whitespace text ->", requests.post(f"{BASE}/documents", json={"title": "x", "text": "   "}).status_code)
print("  empty query     ->", requests.post(f"{BASE}/search", json={"query": ""}).status_code)

pid = ids["Cooking Pasta"]
print(f"\ndelete doc {pid}:", requests.delete(f"{BASE}/documents/{pid}").json())
print("delete again ->", requests.delete(f"{BASE}/documents/{pid}").status_code)
print("final status:", requests.get(f"{BASE}/status").json())