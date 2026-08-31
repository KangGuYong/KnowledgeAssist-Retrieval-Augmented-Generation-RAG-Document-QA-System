# Knowledge Assist RAG - Backend

FastAPI-based backend for the Knowledge Assist RAG application.

## Setup

1. Create virtual environment:

```bash
python -m venv venv
source /app/venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

`paddlepaddle` is pinned to 3.2.2, the newest release with Linux aarch64
wheels on PyPI, so this works on x86_64, aarch64, macOS, and Windows. For the
CUDA build or a newer Paddle on aarch64, install it from PaddlePaddle's own
index instead:

```bash
pip install paddlepaddle==3.3.1 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

3. Configure the local Ollama endpoint and model:

```bash
cp .env.example .env
# The default .env.example uses Ollama at http://192.168.0.169:11434
# with gemma4:26b-a4b-it-q4_K_M.
```

4. Run the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger documentation.

## Directory Structure

- `app/api/` - API routes and request/response models
- `app/services/` - Business logic (RAG, document processing, vector store)
- `app/core/` - Core utilities and dependencies
- `app/storage/` - File and vector database storage

## Key Services

### Document Processor

Handles loading and chunking of PDF, TXT, and DOCX files. PDFs go through the
OCR pipeline below, so text inside images is chunked and embedded like any
other text.

### OCR Service

Wraps PaddleOCR (`korean_PP-OCRv5_mobile_rec` recognition +
`PP-OCRv5_mobile_det` detection) and normalises its output into reading-ordered
text. Models are downloaded on first use and the pipeline is loaded lazily, so
startup is unaffected until the first PDF arrives.

### Vector Store Service

Manages ChromaDB vector database operations.

### RAG Service

Orchestrates the retrieval-augmented generation pipeline using LangChain.

### Parsed Store

Persists MinerU's raw parsed blocks per document (page by page) for the
parsed-result viewer, separately from the chunks that go into the vector
store. See the parsed-result viewer endpoints below.

## PDF parsing (MinerU + PaddleOCR)

PDF layout, table, and equation extraction is delegated to
[MinerU](https://github.com/opendatalab/MinerU), a self-hosted HTTP
service run separately from this backend (`mineru-api`, default
`http://127.0.0.1:8100`, configured via `MINERU_BASE_URL`). Start it with:

```bash
~/mineru-venv/bin/mineru-api --host 0.0.0.0 --port 8100
```

(install with `pip install "mineru[core]"` into its own virtualenv - not
this backend's `app/venv` - MinerU is a separate process, not a Python
dependency of this app). `MINERU_ENABLED=False` skips it entirely and falls
back to plain `pypdf` text extraction, same as when the service is
unreachable or a specific PDF fails to parse: uploads never fail just
because MinerU could not run, they just lose layout/table/equation/image
handling for that document and log a warning.

MinerU returns each page's content as blocks (text, table, equation,
image/chart/etc.) in reading order. Text and headings are kept as-is;
tables come back as HTML, equations as LaTeX. MinerU does not itself read
text embedded inside figures/charts - those blocks are cropped and passed
to PaddleOCR (`korean_PP-OCRv5_mobile_rec` + `PP-OCRv5_mobile_det`), and
the recognised text is spliced back in at the figure's position, labelled
with `OCR_BLOCK_PREFIX` (`[이미지 텍스트]`) so retrieved chunks show where
it came from. `OCR_ENABLED=False` leaves MinerU's own layout/table/equation
extraction running but skips this PaddleOCR augmentation step, so figures
contribute no text.

Chunk metadata carries `ocr_used` and `ocr_image_count` alongside `page`,
so it is visible which answers came from recognised images.

MinerU's raw content_list blocks (unfiltered by the OCR/chunking decisions
above) are also persisted per document under `PARSED_STORAGE_DIR`
(`app/storage/parsed` by default), one JSON file per document, and served
via `GET /api/v1/documents/parsed` (list) and
`GET /api/v1/documents/{document_id}/parsed` (one document's pages/blocks)
for the parsed-result viewer.

OCR runs in a **separate process**. `paddlex` imports `modelscope`, which
imports `torch`, and Paddle's inference predictor segfaults in a process
that has torch loaded - which this one does, for the embedding model. The
worker (`app/services/ocr_worker.py`) is spawned on demand, stubs out
`modelscope` so torch never reaches it, loads the models once, and is
reused for every later page. If it dies or exceeds `OCR_TIMEOUT`, it is
replaced and the upload continues with whatever text the page already had.
Because the worker is spawned (not forked), the entry point that starts the
app must not import torch at import time; the worker refuses to run rather
than crash if it does. Set `OCR_ISOLATE_PROCESS=False` to run OCR in-process
where that conflict does not exist.

On first use the detection and recognition models are downloaded from
Hugging Face into `~/.paddlex/official_models` (a few minutes; the first
upload after a fresh install will block on it). Later runs load from that
cache. On CPU, expect roughly 2 s per image region.

Set `OCR_DEVICE=gpu` (with `paddlepaddle-gpu` installed instead of
`paddlepaddle`) to run recognition on CUDA.

## Default local model configuration

- Embeddings: `nlpai-lab/KURE-v1` (downloaded from Hugging Face on first use)
- Chat LLM: Ollama `gemma4:26b-a4b-it-q4_K_M` at `http://192.168.0.169:11434`
- Vector collection: `documents_kure_v1`

KURE-v1 produces 1,024-dimensional vectors, unlike the previous embedding
model. Upload documents again after this change; vectors in the prior
collection cannot be searched with KURE-v1.

## Testing

Run tests:

```bash
pytest
```

With coverage:

```bash
pytest --cov=app tests/
```
