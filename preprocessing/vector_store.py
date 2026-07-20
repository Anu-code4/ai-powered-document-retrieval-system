import json
import numpy as np
import faiss


# Step 1: Load embedded chunks
with open("embedded_chunks.json", "r", encoding="utf-8") as file:
    embedded_chunks = json.load(file)

print(f"Loaded {len(embedded_chunks)} embedded chunks.")


# Step 2: Extract only embeddings
embeddings = []

for chunk in embedded_chunks:
    embeddings.append(chunk["embedding"])


# Step 3: Convert to NumPy array
embeddings = np.array(embeddings).astype("float32")

print(f"Embedding Matrix Shape: {embeddings.shape}")


# Step 4: Get embedding dimension
dimension = embeddings.shape[1]

print(f"Embedding Dimension: {dimension}")

# Step 5: Create FAISS Index
index = faiss.IndexFlatL2(dimension)

print("FAISS Index Created Successfully.")


# Step 6: Add embeddings
index.add(embeddings)

print(f"Vectors Stored in Index: {index.ntotal}")

# Step 7: Save FAISS index
faiss.write_index(index, "faiss_index.bin")

print("FAISS index saved as faiss_index.bin")


# Finished
print("\n✅ Vector Database Created Successfully!")