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
# Edit .env if needed
```

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

Drag-and-drop file upload component with status indicators.

### ChatWindow

Main chat interface with message history and input.

### Message

Individual message component with markdown support.

### SourceCitation

Displays source document references with metadata.

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
