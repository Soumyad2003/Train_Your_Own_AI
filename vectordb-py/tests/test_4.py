from document_store import DocumentStore

store = DocumentStore()

store.insert_document("Data Structures",
    "A stack is a last-in first-out data structure. You push items onto the top and pop "
    "them off the top. A queue is first-in first-out. Binary trees organise data "
    "hierarchically, and hash tables give average constant-time lookups.")

store.insert_document("Cooking Pasta",
    "To cook pasta, boil water with plenty of salt, add the pasta, and cook until al dente. "
    "Drain it and toss it with sauce straight away. Fresh tomatoes, garlic and olive oil "
    "make a simple and delicious sauce.")

store.insert_document("Photosynthesis",
    "Photosynthesis is how plants convert sunlight, water and carbon dioxide into glucose "
    "and oxygen. It happens in the chloroplasts, where chlorophyll absorbs light energy.")

# A long document, to check that chunking kicks in
long_text = ("The solar system has eight planets orbiting the sun. Mercury is closest, "
             "while Neptune is the farthest. Jupiter is the largest planet. ") * 40
store.insert_document("Solar System", long_text)

print("Documents:")
for d in store.list_documents():
    print(f"  [{d['doc_id']}] {d['title']}: {d['chunks']} chunk(s)")

queries = [
    "How does a stack work?",
    "How do I make a tomato sauce?",
    "How do plants make energy from light?",
    "Which planet is the biggest?",
]
for q in queries:
    print(f"\nQ: {q}")
    for chunk, dist in store.search(q, k=2):
        print(f"   {dist:.3f}  {chunk.title} (chunk {chunk.position})")

# Delete test
removed = store.delete_document(1)
print(f"\nDeleted 'Cooking Pasta' ({removed} chunk). Documents left:",
      [d["title"] for d in store.list_documents()])
print("Sauce query now returns:", store.search("How do I make a tomato sauce?", k=1)[0][0].title)