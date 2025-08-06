# 🤖 Agentic RAG System with GPT‑OSS

Agentic system that performs **Retrieval-Augmented Generation (RAG)** using **OpenAI’s GPT-OSS models** hosted on **Hugging Face** — all accessible through a sleek **Streamlit UI**.

This system dynamically routes queries through retrieval, grading, rewriting, and final answer generation to provide accurate and context-aware answers from uploaded PDF documents.

---

## 🚀 Features

✅ Retrieval-Augmented Generation using GPT-OSS  
✅ Query rewriting for improved semantic understanding  
✅ Document grading to ensure relevance  
✅ OpenAI embeddings support  
✅ Intuitive Streamlit web interface  
✅ 100% Local PDF document processing  
✅ Hugging Face integration (no OpenAI model cost)

---

## 🛠️ Setup Instructions

### 1. 🔧 Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
# or
source venv/bin/activate # On macOS/Linux
```

---

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 How to Use the App

### 4. 🛡️ Configure API Keys

🔑 In the sidebar:

- Enter your **Hugging Face token** (get one from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))  
- Optionally enter your **OpenAI API key** for high-quality embeddings

---

### 5. 📄 Upload & Process Documents

1. Drag and drop one or more **PDF files**
2. Click **🔄 Process Documents** to build your vector-based knowledge base

---

### 6. 💬 Ask Questions

- Use the **chat interface** at the bottom to ask natural language questions about your uploaded documents
- The assistant will intelligently decide whether to retrieve content, rewrite your query, or respond directly

---

## 📚 Tech Stack

| Component        | Description                                      |
|------------------|--------------------------------------------------|
| 🧠 **GPT‑OSS**      | 120B open LLM by OpenAI, hosted by Hugging Face |
| 📘 **LangChain**   | Handles document loading, splitting, and embedding |
| 🕸️ **LangGraph**   | Powers dynamic agentic workflow logic          |
| 💬 **Streamlit**   | Lightweight frontend for chat and interaction |
| 📄 **PDF Loader**  | Ingest and chunk PDF files locally              |

---

## 📝 License

MIT License © 2025 – Built for research and experimentation purposes.

---

## 🧑‍💻 Author

Made with ❤️ by Jenil Soni 
Contributions, feedback, and stars are welcome!
