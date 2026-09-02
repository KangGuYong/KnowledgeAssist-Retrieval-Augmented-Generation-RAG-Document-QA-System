# Knowledge Assist RAG

A fully self-hosted Retrieval-Augmented Generation application for Korean documents.
Upload PDFs, TXT, or DOCX files and ask questions about them.

**No API keys required.** The LLM, the embedding model, the PDF parser, and OCR all
run locally (or on a host you control) — nothing is sent to a third-party API.

For the design rationale behind the pipeline, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Features

- **Korean-first retrieval** — `nlpai-lab/KURE-v1` embeddings, a Korean PaddleOCR
  recognition model, and MinerU configured for Korean layout
- **Layout-aware PDF parsing** — MinerU extracts text, tables, formulas, and figures
  rather than a flat text dump; `pypdf` is the fallback if MinerU is unavailable
- **OCR for figures** — text inside images is recognised and spliced into the page
  under an `[이미지 텍스트]` marker, so charts are searchable
- **Two chunking strategies** — fixed-size character splitting, or semantic chunking
  at embedding-similarity breakpoints
- **Source citations** — every answer cites document, page, relevance score, and the
  figure images the chunk came from
- **Parsed-document viewer** — inspect exactly what MinerU extracted, page by page
- **Per-document search scope** — choose which uploaded documents the chat searches
- **Conversation memory** — follow-up questions are rewritten into standalone queries

## Architecture

### Ingestion

```
PDF upload
  → MinerU /file_parse            (layout, tables, formulas, figures)
  → PaddleOCR on image blocks     (isolated subprocess)
  → drop running headers/footers and page numbers
  → one Document per page
  → merge consecutive pages up to chunk_size
  → split (fixed-size or semantic)
  → KURE-v1 embeddings → ChromaDB
```

The raw MinerU output is also saved to disk as a read-only side channel that backs
the document viewer tab.

### Query

```
question + conversation history
  → LLM call #1: rewrite into a standalone question
  → similarity search (k=10, filtered to the selected documents)
  → LLM call #2: answer from the retrieved chunks
  → answer + source citations
```

## Technology Stack

### Backend

| Area | Technology | Version |
|---|---|---|
| Web framework | FastAPI + Uvicorn | 0.109.0 / 0.27.0 |
| Settings & validation | Pydantic / pydantic-settings | 2.13.5 / 2.15.0 |
| RAG orchestration | LangChain / langchain-community | 1.3.18 / 0.4.2 |
| LLM client | langchain-ollama | 1.1.0 |
| Legacy chain shim | langchain-classic | 1.0.8 |
| Vector store | ChromaDB | 0.4.22 |
| Embeddings | sentence-transformers + `nlpai-lab/KURE-v1` | 3.3.1 |
| LLM | Ollama (`gemma4:26b-a4b-it-q4_K_M`) | — |
| PDF parsing | MinerU (HTTP service) | — |
| OCR | PaddleOCR + PaddlePaddle | 3.7.0 / 3.2.2 |
| PDF fallback | pypdf / PyMuPDF | 4.0.0 / 1.28.2 |
| Tests | pytest | 7.4.4 |

Semantic chunking is a direct port of the "5 Levels of Text Splitting" notebook
(Level 4) rather than `langchain-experimental`, which is therefore not a dependency.

`langchain-classic` is a temporary foothold for the `ConversationalRetrievalChain`
and `ConversationBufferMemory` that LangChain 1.x dropped from the main package.
It goes away once the pipeline is rewritten in LCEL.

### Frontend

React 18.2 · TypeScript 5.3 · Vite 5.0 · axios 1.6 · react-dropzone 14.2 ·
react-markdown 9.0 · DOMPurify 3.4 · lucide-react

No state-management library — `App.tsx` owns the document list and passes it down.

## Prerequisites

- **Python 3.12** (developed against 3.12.3)
- **Node.js 18+**
- **[Ollama](https://ollama.com/)** running, with the chat model pulled
- **[MinerU](https://github.com/opendatalab/MinerU)** running as an HTTP service on
  port 8100 — optional, but without it PDFs fall back to plain text extraction and
  you lose tables, figures, and the document viewer
- **NVIDIA GPU** — optional. The embedding model defaults to `cuda`; set
  `EMBEDDING_DEVICE=cpu` to run without one (noticeably slower).

## Quick Start

### 1. Backend

```bash
cd backend

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

`paddlepaddle` is pinned to 3.2.2, the newest release with Linux aarch64 wheels on
PyPI. For the CUDA build, or a newer Paddle on aarch64, install it from
[PaddlePaddle's own index](https://www.paddlepaddle.org.cn/packages/stable/cpu/).

**Edit `.env` before starting.** `OLLAMA_BASE_URL` defaults to a private LAN address
and will not work on your machine as-is:

```bash
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma4:26b-a4b-it-q4_K_M
```

Then start the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first request loads KURE-v1, which downloads the model on a cold start.

### 2. Frontend

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so no frontend `.env`
is needed unless the backend runs on a different origin.

### 3. Open

Go to **http://localhost:5173**.

## Usage

1. **Upload** — drag PDF, TXT, or DOCX files onto the upload area. Processing a
   large PDF takes a while: MinerU parses it, then every figure is OCR'd, then every
   chunk is embedded. The backend logs `[TIMING]` lines for each stage.
2. **Pick a scope** — use the document selector to limit which documents the chat
   searches. All documents are selected by default.
3. **Ask** — answers cite their sources; expand a citation to see the page, the
   relevance score, and any figures the chunk came from.
4. **Inspect** — the **문서 뷰어** tab shows MinerU's raw parse per page. Table blocks
   show both the extracted HTML and the screenshot MinerU actually parsed, so you can
   compare them.

## API

Interactive docs at **http://localhost:8000/docs**.

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/upload/` | Upload and process one file |
| POST | `/api/v1/upload/batch` | Upload several; one failure does not abort the rest |
| POST | `/api/v1/chat/` | Ask a question, get an answer with sources |
| DELETE | `/api/v1/chat/conversation/{id}` | Clear conversation history |
| GET | `/api/v1/documents/parsed` | List parsed documents — **the effective document list** |
| GET | `/api/v1/documents/{id}/parsed` | One document's parsed blocks, per page |
| GET | `/api/v1/documents/{id}/images/{image_id}` | An extracted figure |
| GET | `/api/v1/documents/` | Stub — always returns `[]` |
| DELETE | `/api/v1/documents/{id}` | Delete the document and every artifact of it |

The upload endpoints accept `chunking_strategy` (`default` or `semantic`),
`chunk_size`, and `chunk_overlap` as form fields.

There is no document-metadata database; the storage directories act as the registry.
That is why `GET /api/v1/documents/` is a stub and the frontend uses
`/api/v1/documents/parsed` instead.

## Configuration

Backend settings live in `backend/.env` (see `.env.example`); defaults are in
`app/config.py`. The ones you are most likely to change:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://192.168.0.169:11434` | **Private LAN default — override this** |
| `LLM_MODEL` | `gemma4:26b-a4b-it-q4_K_M` | Ollama model name |
| `EMBEDDING_MODEL` | `nlpai-lab/KURE-v1` | Changing this needs a new `COLLECTION_NAME` |
| `EMBEDDING_DEVICE` | `cuda` | `cpu`, `cuda`, `cuda:0`, … |
| `RETRIEVAL_K` | `10` | Chunks retrieved per question |
| `RETRIEVAL_REORDER` | `true` | Put the most relevant chunks at both ends of the context |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Also the page-merge target size |
| `CHUNKING_STRATEGY` | `default` | `default` or `semantic` |
| `MAX_DOCUMENTS` | `10` | Total documents allowed at once |
| `MAX_UPLOAD_SIZE` | `10485760` | 10 MB |
| `MINERU_BASE_URL` | `http://127.0.0.1:8100` | Set `MINERU_ENABLED=False` to skip it |
| `OCR_DEVICE` | `cpu` | `cpu`, `gpu`, `gpu:0`, … |
| `OCR_ISOLATE_PROCESS` | `True` | Keep enabled — see below |

Frontend: `VITE_API_BASE_URL` — leave empty to use the Vite dev proxy.

> **`OCR_ISOLATE_PROCESS` should stay `True`.** PaddleOCR segfaults in a process that
> has torch loaded, and this process loads torch for the embedding model. Disabling
> isolation runs OCR in-process and will crash the API.

## Project Structure

```
├── ARCHITECTURE.md              # Design decisions and rationale
├── backend/
│   ├── app/
│   │   ├── api/routes/          # upload, chat, documents
│   │   ├── api/models/          # request/response schemas
│   │   ├── services/
│   │   │   ├── mineru_client.py       # MinerU HTTP client, page assembly
│   │   │   ├── ocr_service.py         # PaddleOCR wrapper + subprocess isolation
│   │   │   ├── ocr_worker.py          # the isolated OCR worker entry point
│   │   │   ├── document_processor.py  # loading, chunking orchestration
│   │   │   ├── chunking.py            # page merging, fixed-size + semantic splitters
│   │   │   ├── parsed_store.py        # persists MinerU output for the viewer
│   │   │   ├── vector_store.py        # ChromaDB + embeddings
│   │   │   └── rag_service.py         # the RAG chain and scoring retriever
│   │   ├── config.py            # Settings
│   │   └── storage/             # uploads, chroma_db, parsed, images
│   ├── tests/                   # 149 tests
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── FileUploader.tsx        # drag-and-drop upload
│       │   ├── DocumentSelector.tsx    # which documents the chat searches
│       │   ├── ChatWindow.tsx          # chat interface
│       │   ├── Message.tsx             # markdown message rendering
│       │   ├── SourceCitation.tsx      # expandable citations with figures
│       │   └── ParsedDocumentViewer.tsx # MinerU parse viewer
│       ├── services/api.ts
│       └── types/
└── docs/superpowers/{specs,plans}/     # design docs and implementation plans
```

## Development

```bash
# Backend
cd backend
pytest                  # 149 tests
black app/

# Frontend
cd frontend
npm run build           # includes tsc type checking
```

`npm run lint` is defined in `package.json`, but no ESLint config file is checked
into the repository, so it currently fails to run.

## Troubleshooting

**Chat fails with a connection error** — Ollama is unreachable. Check
`OLLAMA_BASE_URL`; the default points at a private LAN address.

**PDFs upload but tables and figures are missing** — MinerU is not running, so the
pipeline fell back to plain text extraction. The backend logs a warning naming the
file. Nothing is lost from the upload itself; re-upload once MinerU is up.

**The API process crashes during upload** — check that `OCR_ISOLATE_PROCESS=True`.

**Out of memory loading embeddings** — set `EMBEDDING_DEVICE=cpu`.

**Upload rejected with "Maximum of 10 documents"** — delete a document first, or
raise `MAX_DOCUMENTS`.

**ChromaDB fails to initialise** — delete `backend/app/storage/chroma_db/` and
restart. This discards every indexed document; re-upload them.

**CORS errors** — add your frontend origin to `ALLOWED_ORIGINS`.

## Known Limitations

- **No authentication.** Anyone who can reach the API can read and delete everything.
- **Conversation history is in-memory.** It is lost on restart and is not shared
  across workers.
- **Single instance only.** The OCR worker, conversation memory, and cached model
  singletons are all process-local.
- **No document-metadata database.** The storage directories are the registry.
- **Ten documents at a time**, 10 MB each, by default.
- **Sentence splitting depends on `.`/`?`/`!`.** Documents written without sentence
  terminators do not chunk semantically in any meaningful way.
- **MinerU and Ollama are external processes.** MinerU failures degrade gracefully;
  Ollama being down breaks chat entirely.

## License

MIT — see [LICENSE](LICENSE).
