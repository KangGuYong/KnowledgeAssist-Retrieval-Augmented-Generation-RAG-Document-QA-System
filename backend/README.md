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

## PDF image OCR

Every PDF page is split into layout blocks with PyMuPDF. Text blocks are kept
as they are; each image block is rendered at `OCR_DPI` and passed to PaddleOCR,
and the recognised text is spliced back in at the position the image occupied,
labelled with `OCR_BLOCK_PREFIX` (`[이미지 텍스트]`). The page then flows into
the normal chunking and embedding path.

Two shortcuts keep cost down: a page with almost no native text (a scan) or one
sliced into more than `OCR_MAX_IMAGES_PER_PAGE` fragments is recognised in a
single full-page pass, and repeated images (logos, headers) are recognised once
and reused across the document. Images smaller than `OCR_MIN_IMAGE_SIZE` points
are skipped as decoration.

Chunk metadata carries `ocr_used`, `ocr_image_count`, and `full_page_ocr`
alongside `page`, so it is visible which answers came from recognised images.

OCR runs in a **separate process**. `paddlex` imports `modelscope`, which
imports `torch`, and Paddle's inference predictor segfaults in a process that
has torch loaded — which this one does, for the embedding model. The worker
(`app/services/ocr_worker.py`) is spawned on demand, stubs out `modelscope` so
torch never reaches it, loads the models once, and is reused for every later
page. If it dies or exceeds `OCR_TIMEOUT`, it is replaced and the upload
continues with whatever text the page already had. Because the worker is
spawned (not forked), the entry point that starts the app must not import
torch at import time; the worker refuses to run rather than crash if it does.
Set `OCR_ISOLATE_PROCESS=False` to run OCR in-process where that conflict does
not exist.

On first use the detection and recognition models are downloaded from
Hugging Face into `~/.paddlex/official_models` (a few minutes; the first
upload after a fresh install will block on it). Later runs load from that
cache. On CPU, expect roughly 2 s per image region and 5 s per scanned A4
page.

Set `OCR_DEVICE=gpu` (with `paddlepaddle-gpu` installed instead of
`paddlepaddle`) to run recognition on CUDA, or `OCR_ENABLED=False` to fall back
to plain `pypdf` text extraction. OCR failures never fail an upload: the loader
falls back to text-only extraction and logs a warning.

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
