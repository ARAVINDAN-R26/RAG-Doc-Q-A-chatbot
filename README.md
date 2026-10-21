# 📄 Helpful Chat Agent (RAG with Milvus)

A Streamlit app that lets you upload a document and ask questions about it. The document is split into chunks, embedded locally, and stored in a **Milvus** vector database. At question time, the most relevant chunks are retrieved and passed as context to an OpenAI chat model.

## Features

- **Multi-format upload**: PDF, DOCX, TXT, and CSV
- **Local embeddings** with `all-MiniLM-L6-v2` (384 dimensions), so no embedding API costs
- **Vector search in Milvus** (IVF_FLAT index, L2 distance)
- **Context-grounded answers** from `gpt-4o-mini` (temperature 0), using the top 3 retrieved chunks
- **Simple UI**: upload, ask, read the answer

## How It Works

```
Upload file → Extract text → Chunk → Embed → Store in Milvus
                                                   │
Question → Embed → Top-3 similarity search ────────┘
                         │
                         ▼
              Context + question → GPT-4o-mini → Answer
```

1. **Extraction** (`chatapp.py → read_file`): PDFs use `pdfplumber`, DOCX uses `python-docx`, TXT is read directly, CSV is converted to text with pandas.
2. **Chunking** (`milvusload.py → chunk_text`): `RecursiveCharacterTextSplitter` with 500-character chunks and 100-character overlap.
3. **Indexing** (`milvusload.py`): creates the `rag_docs` collection, embeds each chunk, and inserts the vectors and text.
4. **Retrieval** (`milvussearch.py → search_similar_docs`): embeds the question and returns the text of the 3 nearest chunks.
5. **Answering** (`chatapp.py`): the retrieved chunks go into the system prompt and the question is sent as the user message.

### Milvus collection schema

| Field | Type | Notes |
|-------|------|-------|
| `id` | INT64 | Primary key, auto-generated |
| `embedding` | FLOAT_VECTOR (dim 384) | IVF_FLAT index, `nlist=128`, metric L2 |
| `text` | VARCHAR (max 65535) | The chunk text |

## Project Structure

```
.
├── chatapp.py        # Streamlit UI, file reading, and Q&A flow
├── milvusload.py     # Chunking, collection creation, embedding, insertion, PDF extraction
├── milvussearch.py   # Query embedding and similarity search
├── .env              # OPENAI_API_KEY
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- A running **Milvus** instance on `127.0.0.1:19530` (see below)
- An **OpenAI API key**

### 1. Start Milvus

The quickest way is Milvus Standalone with Docker. Follow the official guide at https://milvus.io/docs/install_standalone-docker.md. Once it is running, it should listen on port `19530`.

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install streamlit pandas python-docx python-dotenv \
            langchain langchain-openai \
            pymilvus sentence-transformers pdfplumber
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
streamlit run chatapp.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The embedding model is downloaded on first run.

## Usage

1. Upload a PDF, DOCX, TXT, or CSV file and wait for the "File indexed successfully" message.
2. Type your question in the text box.
3. Click **Get Answer** to see the response in the answer box.

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Milvus host / port | `milvusload.py`, `milvussearch.py` | `127.0.0.1` / `19530` |
| Collection name | `milvusload.py` | `rag_docs` |
| Chunk size / overlap | `chunk_text()` | 500 / 100 |
| Embedding model | `milvusload.py`, `milvussearch.py` | `all-MiniLM-L6-v2` |
| Chat model | `chatapp.py` | `gpt-4o-mini` |
| Results retrieved | `search_similar_docs(top_k=…)` | 3 |

If you swap the embedding model, update the `dim=384` in the collection schema to match.

## Known Limitations

- **One document at a time**: `create_milvus_collection()` drops the existing `rag_docs` collection on every call, so uploading a new file replaces the previous one.
- **Re-indexing on reruns**: Streamlit reruns the script on every interaction (including clicking **Get Answer**), and the upload block runs each time while a file is present. This means the collection is dropped and rebuilt on every rerun. Consider guarding it with `st.session_state` so each file is indexed only once.
- **Embedding model loaded twice**: `milvussearch.py` reloads `SentenceTransformer` on every query. Reusing the one in `milvusload.py` (or caching it with `st.cache_resource`) would speed up answers.
- **No chat history**: each question is answered independently.
- **No source citations**: retrieved chunks are not shown alongside the answer.
- **Temp files are not cleaned up**: uploaded files are written with `delete=False` and never removed.
- **No handling when Milvus is unreachable**: the app will raise an error if the server isn't running.
- **Possibly outdated imports**: `langchain.schema` and `langchain.text_splitter` are deprecated in recent LangChain versions. Use `langchain_core.messages` and `langchain_text_splitters` if you see warnings.
- **Scanned PDFs**: `pdfplumber` only extracts embedded text; image-only PDFs need OCR.

## Roadmap Ideas

- Support multiple documents with per-file metadata (filename, page number)
- Show the retrieved source chunks under each answer
- Add chat history and follow-up questions
- Stream the LLM response
- Add OCR for scanned PDFs

## Tech Stack

[Streamlit](https://streamlit.io/) · [Milvus](https://milvus.io/) / [pymilvus](https://pypi.org/project/pymilvus/) · [Sentence-Transformers](https://www.sbert.net/) · [LangChain](https://python.langchain.com/) · [OpenAI](https://platform.openai.com/) · [pdfplumber](https://github.com/jsvine/pdfplumber) · [python-docx](https://python-docx.readthedocs.io/) · [pandas](https://pandas.pydata.org/)
