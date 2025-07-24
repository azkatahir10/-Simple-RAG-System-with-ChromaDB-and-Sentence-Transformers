from sentence_transformers import SentenceTransformer

def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(model, documents):
    return model.encode(documents).tolist()

def add_documents_to_collection(collection, documents, embeddings):
    for doc, emb in zip(documents, embeddings):
        collection.add(
            documents=[doc],
            embeddings=[emb],
            ids=[str(hash(doc))]
        )
