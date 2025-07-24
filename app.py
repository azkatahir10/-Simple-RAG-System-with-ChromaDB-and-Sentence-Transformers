import streamlit as st
from faq_data import faq_documents
from utils import get_embedding_model, embed_documents, add_documents_to_collection
import chromadb

# Initialize session state
if "collection" not in st.session_state:
    # Embed documents and initialize collection once
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="python_faqs")
    model = get_embedding_model()
    document_embeddings = embed_documents(model, faq_documents)
    add_documents_to_collection(collection, faq_documents, document_embeddings)
    st.session_state.collection = collection
    st.session_state.model = model

# Streamlit UI
st.set_page_config(page_title="RAG FAQ System", layout="centered")
st.title("🧠 RAG FAQ Chatbot")
st.markdown("Ask me something about Python and virtual environments!")

query = st.text_input("🔍 Enter your question here:")

if query:
    # Embed query and search
    query_embedding = st.session_state.model.encode(query).tolist()
    results = st.session_state.collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    retrieved_doc = results["documents"][0][0]
    distance = results["distances"][0][0]

    # Display results
    st.subheader("📄 Retrieved Context")
    st.write(retrieved_doc)

    st.subheader("💡 Answer")
    st.markdown(f"**Q:** {query}  \n**A:** {retrieved_doc}")

    st.caption(f"Distance: {distance:.4f} (lower = better match)")
