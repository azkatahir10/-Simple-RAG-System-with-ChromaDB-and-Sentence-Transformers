# 🧠 RAG FAQ System

This is a simple Retrieval-Augmented Generation (RAG) application built with **ChromaDB**, **Hugging Face Transformers**, and **Streamlit** to answer frequently asked questions (FAQs) using contextually relevant documents.

---

## 🚀 Features

- 🧾 Upload documents (`.txt`) as context.
- 🤖 Ask questions based on the uploaded documents.
- 🔍 Uses sentence-transformer embeddings for similarity search.
- 📦 Local vector storage with ChromaDB.
- 💡 Streamlit UI for interactive querying.

---

## 📂 Project Structure

```

rag\_faq\_system/
│
├── app.py              # Streamlit interface
├── rag\_app.py          # Core RAG logic
├── config.py           # Configuration (model names, db paths)
├── utils/
│   └── helper.py       # Utility functions (embedding, retrieval, response)
├── data/
│   └── your\_docs.txt   # Text documents to embed
├── venv/               # Python virtual environment
└── requirements.txt    # Python dependencies

````

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-faq-system.git
cd rag-faq-system
````

### 2. Create & activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 How it works

1. Documents are embedded using `sentence-transformers` models.
2. ChromaDB stores the vector representations.
3. User inputs a question via Streamlit.
4. Similar documents are retrieved using cosine similarity.
5. HuggingFace `transformers` generates a response using context + query.

---

## 📌 Dependencies

* `sentence-transformers`
* `transformers`
* `chromadb`
* `streamlit`

> All packages listed in `requirements.txt`

---

## 🛡️ License

MIT License – feel free to use, modify, and distribute.

