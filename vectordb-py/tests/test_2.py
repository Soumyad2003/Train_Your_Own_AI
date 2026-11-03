from chunker import chunk_text

# 600 fake words: w0 w1 w2 ... w599
text = " ".join(f"w{i}" for i in range(600))
chunks = chunk_text(text, chunk_size=250, overlap=50)

print("Number of chunks:", len(chunks))
for i, c in enumerate(chunks):
    words = c.split()
    print(f"chunk {i}: {len(words)} words, {words[0]} ... {words[-1]}")

# Edge cases
print("Empty text:", chunk_text(""))
print("Short text:", len(chunk_text("just a few words here")), "chunk")