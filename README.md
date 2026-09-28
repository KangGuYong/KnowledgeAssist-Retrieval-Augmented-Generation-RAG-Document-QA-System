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


| Area                  | Technology                                 | Version          |
| ----------------------- | -------------------------------------------- | ------------------ |
| Web framework         | FastAPI + Uvicorn                          | 0.109.0 / 0.27.0 |
| Settings & validation | Pydantic / pydantic-settings               | 2.13.5 / 2.15.0  |
| RAG orchestration     | LangChain / langchain-community            | 1.3.18 / 0.4.2   |
| LLM client            | langchain-ollama                           | 1.1.0            |
| Vector store          | ChromaDB                                   | 0.4.22           |
| Embeddings            | sentence-transformers + `nlpai-lab/KURE-v1` | 3.3.1            |
| LLM                   | Ollama (`gemma4:26b-a4b-it-q4_K_M`)        | —               |
| PDF parsing           | MinerU (HTTP service)                      | —               |
| OCR                   | PaddleOCR + PaddlePaddle                   | 3.7.0 / 3.2.2    |
| PDF fallback          | pypdf / PyMuPDF                            | 4.0.0 / 1.28.2   |
| Tests                 | pytest                                     | 7.4.4            |

Semantic chunking is a direct port of the "5 Levels of Text Splitting" notebook
(Level 4) rather than `langchain-experimental`, which is therefore not a dependency.

`langchain-classic` is not imported by this project. It holds the legacy chains
LangChain 1.x dropped from the main package and served as a migration foothold;
the LCEL rewrite removed it, and it now only rides along as a transitive
dependency of `langchain-community`.

### Frontend

React 18.2 · TypeScript 5.3 · Vite 5.0 · axios 1.6 · react-dropzone 14.2 ·
react-markdown 9.0 · DOMPurify 3.4 · lucide-react

No state-management library — `App.tsx` owns the document list and passes it down.

## Prerequisites

- **Python 3.12** (developed against 3.12.3)
- **Node.js 18+**
- **[Ollama](https://ollama.com/)** running, with the chat model pulled
- **[MinerU](https://github.com/opendatalab/MinerU) 3.x** (`mineru-api`) — optional,
  but without it PDFs fall back to plain text extraction and you lose tables,
  figures, and the document viewer
- **NVIDIA GPU** — optional. The embedding model and the reranker default to `cuda`;
  set `EMBEDDING_DEVICE=cpu` and `RERANK_DEVICE=cpu` to run without one (noticeably
  slower).

## Quick Start

Four processes take part. Start them in this order, each in its own terminal:


| Process           | Default address          | Who points at it                     |
| ------------------- | -------------------------- | -------------------------------------- |
| Ollama            | `http://localhost:11434` | backend `.env` → `OLLAMA_BASE_URL`   |
| MinerU            | `http://127.0.0.1:8100`  | backend `.env` → `MINERU_BASE_URL`   |
| Backend (uvicorn) | `http://127.0.0.1:8000`  | frontend `.env` → `API_PROXY_TARGET` |
| Frontend (Vite)   | `http://localhost:5173`  | your browser                         |

**Check that ports 8000, 8100, and 5173 are free first** (`ss -ltnp | grep -E ':(8000|8100|5173)\b'`). If something else already listens on 8000, the backend
fails to start, or worse, the frontend proxy silently talks to that other server
and every API call comes back `404 {"detail":"Not Found"}`. Run the backend on
another port instead — see step 3.

### 1. Ollama

```bash
ollama pull gemma4:26b-a4b-it-q4_K_M
curl http://localhost:11434/api/tags        # the model must be listed
```

### 2. MinerU (optional)

The backend calls MinerU's synchronous `POST /file_parse` endpoint, which is what
`mineru-api` from the `mineru` package serves (verified against 3.4.5). Other
MinerU servers, such as the job-based `/v1/parse/jobs` API, are not compatible.

MinerU pins its own torch, so give it its own virtual environment:

```bash
python3.12 -m venv ~/mineru-venv
~/mineru-venv/bin/pip install "mineru[pipeline]==3.4.5"
~/mineru-venv/bin/mineru-api --host 127.0.0.1 --port 8100
```

Check it is the right server — this must print `/file_parse`:

```bash
curl -s http://127.0.0.1:8100/openapi.json | grep -o '"/file_parse"'
```

The first parse downloads MinerU's layout models, so it is slow. To skip MinerU
entirely, set `MINERU_ENABLED=False` in the backend `.env`.

### 3. Backend

```bash
cd backend

python3.12 -m venv venv
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
MINERU_BASE_URL=http://127.0.0.1:8100       # or MINERU_ENABLED=False
# No GPU? Also set EMBEDDING_DEVICE=cpu and RERANK_DEVICE=cpu
```

Then start the server **from the `backend/` directory** — `.env` and the
`app/storage/` paths are resolved relative to it:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Startup loads KURE-v1 (downloaded on a cold start), so wait for
`Application startup complete.` The reranker loads on the first question. Check:

```bash
curl http://127.0.0.1:8000/health           # {"status":"healthy"}
```

**If port 8000 is taken**, pick another one, e.g. `--port 8001`, and set
`API_PROXY_TARGET=http://127.0.0.1:8001` in `frontend/.env` in the next step.

### 4. Frontend

```bash
cd frontend
npm install
cp .env.example .env              # only needed to change API_PROXY_TARGET
npm run dev
```

The Vite dev server proxies `/api` to `API_PROXY_TARGET` (default
`http://127.0.0.1:8000`), so the browser only ever talks to port 5173 and CORS
does not come into play. Leave `VITE_API_BASE_URL` empty: setting it makes the
browser call the backend directly, and that origin then has to be listed in the
backend's `ALLOWED_ORIGINS`. Vite reads `.env` only at startup — restart
`npm run dev` after editing it.

Check that the proxy reaches the backend — this must return a JSON list (`[]` on a
fresh install), not `{"detail":"Not Found"}`:

```bash
curl http://127.0.0.1:5173/api/v1/documents/parsed
```

### 5. Open

Go to **http://localhost:5173**. Vite listens on every interface, so other machines
can use `http://<host-ip>:5173`.

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


| Method | Path                                       | Description                                             |
| -------- | -------------------------------------------- | --------------------------------------------------------- |
| POST   | `/api/v1/upload/`                          | Upload and process one file                             |
| POST   | `/api/v1/upload/batch`                     | Upload several; one failure does not abort the rest     |
| POST   | `/api/v1/chat/`                            | Ask a question, get an answer with sources              |
| DELETE | `/api/v1/chat/conversation/{id}`           | Clear conversation history                              |
| GET    | `/api/v1/documents/parsed`                 | List parsed documents — **the effective document list** |
| GET    | `/api/v1/documents/{id}/parsed`            | One document's parsed blocks, per page                  |
| GET    | `/api/v1/documents/{id}/images/{image_id}` | An extracted figure                                     |
| GET    | `/api/v1/documents/`                       | Stub — always returns `[]`                              |
| DELETE | `/api/v1/documents/{id}`                   | Delete the document and every artifact of it            |

The upload endpoints accept `chunking_strategy` (`default` or `semantic`),
`chunk_size`, and `chunk_overlap` as form fields.

There is no document-metadata database; the storage directories act as the registry.
That is why `GET /api/v1/documents/` is a stub and the frontend uses
`/api/v1/documents/parsed` instead.

## Configuration

Backend settings live in `backend/.env` (see `.env.example`); defaults are in
`app/config.py`. The ones you are most likely to change:


| Variable                       | Default                           | Description                                                            |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------------------ |
| `OLLAMA_BASE_URL`              | `http://192.168.0.169:11434`      | **Private LAN default — override this**                               |
| `LLM_MODEL`                    | `gemma4:26b-a4b-it-q4_K_M`        | Ollama model name                                                      |
| `EMBEDDING_MODEL`              | `nlpai-lab/KURE-v1`               | Changing this needs a new `COLLECTION_NAME`                             |
| `EMBEDDING_DEVICE`             | `cuda`                            | `cpu`, `cuda`, `cuda:0`, …                                            |
| `RETRIEVAL_K`                  | `10`                              | Chunks retrieved per question                                          |
| `RETRIEVAL_REORDER`            | `true`                            | Put the most relevant chunks at both ends of the context               |
| `RERANK_ENABLED`               | `true`                            | Rescore a wider candidate pool with a cross-encoder, keep `RETRIEVAL_K` |
| `RERANK_MODEL`                 | `dragonkue/bge-reranker-v2-m3-ko` | Korean-tuned multilingual reranker (2.27GB)                            |
| `RERANK_DEVICE`                | `cuda`                            | `cpu`, `cuda`, `cuda:0`, …                                            |
| `RERANK_CANDIDATE_K`           | `30`                              | Candidates fetched before reranking                                    |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200`                    | Also the page-merge target size                                        |
| `CHUNKING_STRATEGY`            | `default`                         | `default` or `semantic`                                                |
| `MAX_DOCUMENTS`                | `10`                              | Total documents allowed at once                                        |
| `MAX_UPLOAD_SIZE`              | `10485760`                        | 10 MB                                                                  |
| `MINERU_BASE_URL`              | `http://127.0.0.1:8100`           | Set `MINERU_ENABLED=False` to skip it                                   |
| `OCR_DEVICE`                   | `cpu`                             | `cpu`, `gpu`, `gpu:0`, …                                              |
| `OCR_ISOLATE_PROCESS`          | `True`                            | Keep enabled — see below                                              |

Frontend settings live in `frontend/.env` (see `.env.example`): `API_PROXY_TARGET`
is where the Vite dev proxy forwards `/api` (default `http://127.0.0.1:8000`), and
`VITE_API_BASE_URL` should stay empty so requests go through that proxy.

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
│   ├── tests/                   # 200 tests
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
pytest                  # 200 tests
black app/

# Frontend
cd frontend
npm run build           # includes tsc type checking
```

`npm run lint` is defined in `package.json`, but no ESLint config file is checked
into the repository, so it currently fails to run.

## Troubleshooting

**Every API call returns `404 {"detail":"Not Found"}`, or the document list stays
empty** — the frontend proxy is reaching some other server. Check what owns the
backend port (`ss -ltnp | grep :8000`); `curl http://127.0.0.1:8000/` must answer
with `"name":"Knowledge Assist RAG API"`. Move the backend to a free port and set
`API_PROXY_TARGET` accordingly, and make sure `VITE_API_BASE_URL` in `frontend/.env`
is empty.

**Upload fails with `Expected IDs to be a non-empty list, got []`** — no text was
extracted, so there was nothing to index. MinerU parses rendered page images, so a
PDF whose fonts are not embedded can come out blank. The failed upload still leaves
an empty entry in the document list; delete it.

**Chat fails with a connection error** — Ollama is unreachable. Check
`OLLAMA_BASE_URL`; the default points at a private LAN address.

**PDFs upload but tables and figures are missing** — MinerU is not reachable at
`MINERU_BASE_URL` (or it is not a `/file_parse` server — see Quick Start step 2), so
the pipeline fell back to plain text extraction. The backend logs a warning naming the
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
