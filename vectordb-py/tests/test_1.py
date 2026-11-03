from ollama_client import OllamaClient

ollama = OllamaClient()
print("Ollama online:", ollama.is_available())

vec = ollama.embed("A binary tree is a hierarchical data structure.")
print("Embedding dims:", len(vec))
print("First 5 values:", vec[:5])

print(ollama.generate("In one sentence, what is a vector database?"))