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

questions = [
    "What is the difference between a stack and a queue?",   # answerable
    "How do I cook pasta properly?",                         # answerable
    "Who won the football World Cup in 2018?",               # NOT in the documents
]

for q in questions:
    print(f"\nQ: {q}")
    result = store.ask(q)
    print("A:", result["answer"])
    print("Sources:", result["sources"])