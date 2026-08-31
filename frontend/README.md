# Knowledge Assist RAG - Frontend

React + TypeScript frontend for the Knowledge Assist RAG application.

## Setup

1. Install dependencies:

```bash
npm install
```

2. Configure environment (optional):

```bash
cp .env.example .env
```

`VITE_API_BASE_URL` is empty by default, so requests go through Vite's dev
proxy (`/api` -> `http://localhost:8000`, no CORS involved). Set it only if
the backend runs on a different origin.

3. Start development server:

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

## Build for Production

```bash
npm run build
```

The production build will be in the `dist/` folder.

## Project Structure

- `src/components/` - React components
- `src/services/` - API client
- `src/types/` - TypeScript type definitions
- `src/styles/` - CSS files

## Key Components

### FileUploader

Drag-and-drop file upload component with status indicators. An "고급 설정"
(advanced settings) panel lets you pick the chunking strategy per upload
(`default` character-based, or `semantic` embedding-similarity based) and,
for `default`, override the chunk size/overlap - see the backend README's
Document Processor section for what each strategy does.

### ChatWindow

Main chat interface with message history and input.

### Message

Individual message component with markdown support.

### SourceCitation

Displays source document references with metadata: a relevance percentage
(from the response's `similarity_score`), and inline thumbnails for any
diagrams the answer's text was recognised from (`image_urls`) - click a
thumbnail to view it full-size, since OCR'd text can misread details the
original image makes clear.

### ParsedDocumentViewer

Renders MinerU's raw parsed blocks (page by page) for a selected document,
reachable via the "문서 뷰어" tab.

## Development

Type checking:

```bash
npm run build
```

Linting:

```bash
npm run lint
```
