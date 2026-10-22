# 🤖 Langchain_Trail — Tool-Calling RAG with MultiQuery & Reranking

A modular LangChain project where an LLM (served by **Groq**) decides when to call a **RAG tool**. The tool loads a PDF into a persistent **ChromaDB** vector store, then answers questions in four steps: it expands the query with **MultiQuery**, retrieves candidate chunks, **reranks** them with a cross-encoder, and generates an answer strictly from the best chunks.

---

## 🗂️ Project Structure

```
Langchain_Trail/
├── Data/
│   └── Personal_details.pdf     # Source document for RAG
├── Tools/
│   ├── RAG/
│   │   ├── Embeddings.py        # HuggingFace embedding wrapper
│   │   ├── ingestion.py         # PDF loading + text chunking
│   │   ├── VectorDB.py          # ChromaDB wrapper (dedupe, search, delete)
│   │   ├── ReRanker.py          # MultiQuery retrieval + cross-encoder reranking
│   │   └── RAGTool.py           # RAG class: ingest + retrieve + generate
│   ├── tools.py                 # RAG_Tools: the LangChain tool the LLM can call
│   └── Web/                     # Reserved for a future web-search tool
├── chroma/                      # Persisted ChromaDB vector store
├── config.py                    # Models, retrieval settings, shared vector store
├── main.py                      # Entry point: LLM with the bound RAG tool
├── requirements.txt
├── .env                         # API keys (not committed)
├── .gitignore
└── LICENSE
```

---

## ⚙️ How It Works

**Question flow**

```
User question
     │
     ▼
ChatGroq (openai/gpt-oss-20b) with bound tool RAG_Tools
     │   decides to call the tool
     ▼
RAG_Tools(query)  ──►  RAG.invoke(query)
                            │
                            ▼
              MultiQueryRetriever
              (LLM writes alternative phrasings of the query)
                            │
                            ▼
              Chroma similarity search, k=5 per phrasing
                            │
                            ▼
              CrossEncoderReranker
              (rescores every (query, chunk) pair, keeps the top 4)
                            │
                            ▼
              Prompt + ChatGroq + StrOutputParser
                            │
                            ▼
                      Final answer
```

**Ingestion (run once per document)**

```
PDF ──► PyMuPDFLoader ──► RecursiveCharacterTextSplitter ──► HuggingFace Embeddings ──► ChromaDB
```

1. `main.py` binds `RAG_Tools` to the Groq model with `bind_tools`.
2. For a document question the model returns a tool call.
3. `RAG_Tools` runs the RAG pipeline: MultiQuery, retrieval, rerank, then answer.
4. If the retrieved context isn't enough, the system prompt makes the model say it doesn't have enough context.

---

## 🤖 Models Used

| Component | Model | Set in |
|---|---|---|
| LLM (tool calling, MultiQuery, answer) | `openai/gpt-oss-20b` via Groq, reasoning effort `low` | `config.py` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | `Tools/RAG/Embeddings.py` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `config.py` |
| Vector store | ChromaDB, collection `knowledge` | `config.py` |

---

## 🚀 Getting Started

### 1. Create a virtual environment and install

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Ingest the document

Put your PDF in `Data/`, set its path at the bottom of `Tools/RAG/RAGTool.py`, then run from the project root:

```bash
python -m Tools.RAG.RAGTool
```

This loads the PDF, splits it into chunks, embeds them, and stores them in `chroma/`. It prints the chunk count and a test answer.

> **First time after changing chunk settings:** delete the `chroma/` folder before ingesting, so no old chunks remain.

### 4. Run

```bash
python main.py
```

It prints the startup time, the answer, and the answer time, so you can see where latency goes.

---

## 💬 Example

```python
from Tools.tools import RAG_Tools

print(RAG_Tools.invoke({'query': 'Who is Karthik brother'}))
```

Or use the RAG class directly:

```python
from Tools.RAG.RAGTool import RAG

rag = RAG()
print(rag.invoke('Where does Karthik work?'))
```

---

## 🧩 Module Reference

| Module | Class / function | Responsibility |
|---|---|---|
| `config.py` | `Config`, `get_vector_db()` | Settings, the Groq model, and one shared vector store created on first use |
| `Tools/RAG/Embeddings.py` | `Embeddings` | Wraps `HuggingFaceEmbeddings` |
| `Tools/RAG/ingestion.py` | `Ingestion` | `Load_Data(path)` loads a PDF; `split_chunks(data)` splits it |
| `Tools/RAG/VectorDB.py` | `VectorData` | `add_docs` (deduplicated), `remove_source`, `similarity_search`, `as_retriever`, `remove_data` |
| `Tools/RAG/ReRanker.py` | `ReRanker` | `rank(query)`: MultiQuery retrieval plus cross-encoder reranking |
| `Tools/RAG/RAGTool.py` | `RAG` | `PipeLine(path)` ingests a PDF; `invoke(query)` retrieves and answers |
| `Tools/tools.py` | `RAG_Tools`, `get_rag()` | The tool the LLM calls; `get_rag()` builds the pipeline once and reuses it |

---

## 🔍 Key Settings

| Setting | Value | Where |
|---|---|---|
| `model_name` | `openai/gpt-oss-20b` | `config.py` |
| `reasoning_effort` | `low` (`medium` / `high` give slower, deeper reasoning) | `config.py` |
| `reranker_model_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `config.py` |
| `retrieval_k` | 5 chunks per query phrasing | `config.py` |
| `rerank_top_n` | 4 chunks kept after reranking | `config.py` |
| `chunk_size` / `chunk_overlap` | 500 / 50 | `Tools/RAG/ingestion.py` |
| Separators | paragraph, line, sentence, word | `Tools/RAG/ingestion.py` |
| Vector store directory | `chroma/` (absolute path from `config.py`) | `config.py` |

---

## 📝 Notes

- **No duplicate chunks.** Each chunk gets a deterministic ID, and `PipeLine()` deletes the file's old chunks before adding new ones, so re-running ingestion is safe.
- **Heavy objects load once.** `get_rag()` caches the pipeline, and `main.py` calls it at startup. Importing `config.py` no longer loads any model.
- **Offline mode.** After the models are cached you can set `HF_HUB_OFFLINE=1` in `.env` to skip Hugging Face network checks. If you switch to a model that isn't cached yet (for example a different reranker), remove that setting for one run so the model can download.
- **Latency.** A question makes three sequential LLM calls (decide the tool, MultiQuery, answer) plus a CPU rerank step. Lowering `reasoning_effort` and using the small reranker are the main speed levers.
- `main.py` handles the tool call and prints its result; it does not yet send the result back to the model for a final reply.
- `Tools/Web/` is empty and reserved for a future web-search tool.

---

## 🛠️ Roadmap

- Rebuild the flow as a LangGraph graph (query rewrite, routing, cache, memory)
- Per-user memory and a semantic cache
- FastAPI service, then an HTML/CSS chat page
- Web-search tool under `Tools/Web/`
- Test question set to measure retrieval quality

---

## 📄 License

See the [LICENSE](LICENSE) file.
