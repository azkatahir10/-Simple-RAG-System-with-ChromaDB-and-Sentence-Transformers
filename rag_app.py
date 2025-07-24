from faq_data import faq_documents
from utils import get_embedding_model, embed_documents, add_documents_to_collection
import chromadb

print("🟢 Starting RAG App..")
print("[⏳] Embedding documents...")

# --- Step 1: Initialize ChromaDB ---
client = chromadb.EphemeralClient()  # Use PersistentClient(path=".chromadb") for persistence
collection = client.get_or_create_collection(name="python_faqs")

# --- Step 2: Load Embedding Model ---
model = get_embedding_model()

# --- Step 3: Embed and Add to Collection ---
document_embeddings = embed_documents(model, faq_documents)
add_documents_to_collection(collection, faq_documents, document_embeddings)

print("[✓] All FAQ documents embedded and added to ChromaDB.")

# --- Step 4: Query ---
query_text = "How do I make a virtual environment?"
query_embedding = model.encode(query_text).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

# --- Step 5: Display Results ---
print("\n[Query]")
print(query_text)

retrieved_doc = results["documents"][0][0]
distance = results["distances"][0][0]
print("\n[Retrieved Document]")
print(retrieved_doc)
print(f"\n[Distance]: {distance:.4f}")

# --- Step 6: Simulated LLM Prompt ---
llm_prompt = f"Based on the following context, please answer the question.\nContext: {retrieved_doc}\n\nQuestion: {query_text}"
print("\n[Conceptual LLM Prompt]")
print(llm_prompt)

print("\n✅ Script completed.")
