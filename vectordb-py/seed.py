import argparse
import sys

import requests

DOCS = {
    "Python Basics":
        "Python is a high-level programming language known for its readable syntax. A list is an "
        "ordered, mutable collection, so you can add, remove or change items after creating it. A "
        "tuple is also ordered but immutable, which means it cannot be modified once created. "
        "Dictionaries store key-value pairs and give fast lookups by key. Python uses indentation "
        "to define blocks of code instead of curly braces. Common uses include data analysis, "
        "automation scripts, web backends with frameworks like FastAPI and Django, and machine "
        "learning.",

    "JavaScript and the Browser":
        "JavaScript is the programming language of the web browser. It lets developers make pages "
        "interactive by responding to clicks, updating the page without reloading, and talking to "
        "servers with fetch requests. Node.js allows JavaScript to run outside the browser, for "
        "example in a backend server built with Express. React is a library for building user "
        "interfaces from reusable components, and it keeps the screen in sync with application "
        "state. Variables can be declared with let and const, and functions can be written as "
        "arrow functions.",

    "Masala Chai":
        "To make masala chai, bring one cup of water to a boil with a few crushed cardamom pods, a "
        "small piece of ginger and a cinnamon stick. Add two teaspoons of black tea leaves and let "
        "the mixture simmer for about two minutes. Pour in one cup of whole milk, add sugar to "
        "taste, and simmer for another three minutes until the tea turns a deep golden brown. "
        "Strain it into cups and serve hot. Using fresh ginger and freshly crushed cardamom gives "
        "the strongest flavour.",

    "Photosynthesis":
        "Photosynthesis is the process plants use to make their own food. Leaves absorb sunlight "
        "using chlorophyll, the green pigment inside chloroplasts. The plant combines carbon "
        "dioxide from the air with water from the soil to produce glucose, a sugar used for energy "
        "and growth, and releases oxygen as a by-product. Without photosynthesis, the oxygen in "
        "our atmosphere would run out and most food chains would collapse.",

    "The Water Cycle":
        "The water cycle describes how water moves around the Earth. Heat from the sun causes "
        "water from oceans, rivers and lakes to evaporate into vapour. As the vapour rises and "
        "cools it condenses into tiny droplets that form clouds. When the droplets grow heavy "
        "enough they fall as rain, snow or hail, which is called precipitation. The water then "
        "collects in rivers, lakes and underground, and the cycle begins again.",

    "Vector Databases Explained":
        "A vector database stores data as high-dimensional vectors, also called embeddings, and "
        "finds items by similarity instead of by exact keyword match. An embedding model turns a "
        "piece of text, an image or a sound clip into a list of numbers, typically several hundred "
        "of them. Items with similar meaning end up close together in this space, so a question "
        "about feeling sleepy after lunch can match a passage about the afternoon energy slump "
        "even though they share no words. The closeness between two vectors is measured with a "
        "distance metric. Cosine distance compares the direction of two vectors and ignores their "
        "length, which makes it the most common choice for text. Euclidean distance measures the "
        "straight line between two points, and Manhattan distance adds up the differences along "
        "each dimension.\n\n"
        "The simplest way to search is brute force, which compares the query against every stored "
        "vector. It is always exact, but its cost grows linearly with the number of vectors, so it "
        "becomes slow for millions of items. Tree structures such as the KD-Tree work well in two "
        "or three dimensions, but in hundreds of dimensions they stop pruning and end up visiting "
        "nearly every point. This problem is known as the curse of dimensionality.\n\n"
        "HNSW, short for Hierarchical Navigable Small World, solves this with a layered graph. "
        "Every vector lives in the bottom layer, while a random subset also appears in the layers "
        "above, forming sparse express lanes. A search enters at the top layer, hops greedily "
        "toward the query, and drops down a layer at a time until it reaches the dense bottom "
        "layer, where it explores more widely. The parameter M controls how many links each node "
        "keeps, and a larger M improves accuracy at the cost of memory. The parameter "
        "ef_construction sets how many candidates are considered while building the graph, and "
        "ef_search sets how many are considered while querying. Raising ef_search increases recall "
        "but makes each query slower. HNSW is approximate, so its quality is measured as recall: "
        "the fraction of the true nearest neighbours that it actually returns.\n\n"
        "Retrieval augmented generation, or RAG, combines a vector database with a language model. "
        "The documents are split into chunks, each chunk is embedded and indexed, and at question "
        "time the closest chunks are retrieved and placed into the prompt so the model can answer "
        "from them rather than from memory.",
}


def main():
    p = argparse.ArgumentParser(description="Load the sample documents into the running API.")
    p.add_argument("--reset", action="store_true", help="delete every existing document first")
    p.add_argument("--base", default="http://127.0.0.1:8000", help="API address")
    args = p.parse_args()

    try:
        status = requests.get(f"{args.base}/status", timeout=5).json()
    except requests.RequestException:
        sys.exit("Can't reach the API. Is uvicorn running?")
    if not status["ollama"]:
        sys.exit("The API is up, but Ollama is offline. Start Ollama and try again.")

    existing = requests.get(f"{args.base}/documents").json()
    if args.reset:
        for d in existing:
            requests.delete(f"{args.base}/documents/{d['doc_id']}")
        print(f"deleted {len(existing)} existing document(s)")
        existing = []
    have = {d["title"] for d in existing}

    for title, text in DOCS.items():
        if title in have:
            print(f"skip   {title} (already loaded)")
            continue
        r = requests.post(f"{args.base}/documents", json={"title": title, "text": text}, timeout=300)
        if r.status_code != 201:
            sys.exit(f"failed {title}: {r.status_code} {r.json()}")
        print(f"added  {title} ({r.json()['chunks']} chunk(s))")

    s = requests.get(f"{args.base}/status").json()
    print(f"\nDone: {s['documents']} documents, {s['chunks']} chunks.")


if __name__ == "__main__":
    main()