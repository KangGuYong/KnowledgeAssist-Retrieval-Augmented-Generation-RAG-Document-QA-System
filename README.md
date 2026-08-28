# Knowledge Assist RAG

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about them using AI. The system uses LangChain, ChromaDB for vector storage, and supports both Anthropic and OpenAI LLMs.

## ✨ Features

- **Document Upload**: Upload PDFs, TXT, and DOCX files with drag-and-drop support
- **Smart Chunking**: Automatically splits documents into optimal chunks for retrieval
- **Vector Search**: Uses ChromaDB for fast semantic search across documents
- **AI-Powered Answers**: Leverages Claude or GPT models for context-aware responses
- **Source Citations**: Every answer includes references to source documents with page numbers
- **Conversation Memory**: Maintains conversation context for follow-up questions
- **Beautiful UI**: Modern React interface with real-time updates

## 🏗️ Architecture

### RAG Pipeline Flow

```
1. Upload Document
   ↓
2. Parse & Chunk (with overlap)
   ↓
3. Generate Embeddings (Sentence Transformers)
   ↓
4. Store in Vector DB (ChromaDB)
   ↓
5. User Question → Semantic Search
   ↓
6. Retrieve Relevant Chunks
   ↓
7. LLM Context + Question → Answer
   ↓
8. Return Answer + Source Citations
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** 0.109.0 - High-performance Python web framework
- **LangChain** 0.1.0 - RAG pipeline orchestration
- **ChromaDB** 0.4.22 - Vector database for embeddings
- **Sentence Transformers** 2.2.2 - Local embedding generation
- **Pydantic** 2.5.0 - Data validation and settings
- **Anthropic/OpenAI** - LLM providers

### Frontend
- **React** 18.2.0 - Modern UI library
- **TypeScript** 5.3.3 - Type-safe development
- **Vite** 5.0.11 - Fast build tool
- **Axios** 1.6.5 - HTTP client
- **React Dropzone** 14.2.3 - File upload with drag & drop
- **React Markdown** 9.0.1 - Render formatted responses
- **Lucide React** - Beautiful icons

## 📁 Project Structure

```
KnowledgeAssist RAG/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/        # API endpoints (upload, chat, documents)
│   │   │   └── models/        # Request/response models
│   │   ├── services/
│   │   │   ├── vector_store.py        # ChromaDB integration
│   │   │   ├── document_processor.py  # File parsing & chunking
│   │   │   └── rag_service.py         # RAG pipeline
│   │   ├── core/              # Configuration
│   │   └── storage/           # File and vector storage
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUploader.tsx      # Drag-and-drop upload
│   │   │   ├── ChatWindow.tsx        # Chat interface
│   │   │   ├── Message.tsx           # Message display
│   │   │   └── SourceCitation.tsx    # Source references
│   │   ├── services/          # API client
│   │   ├── types/             # TypeScript definitions
│   │   └── styles/            # CSS files
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:
- **Python 3.9+** installed
- **Node.js 18+** installed
- An API key from [Anthropic](https://console.anthropic.com/) or [OpenAI](https://platform.openai.com/)

### Step 1: Backend Setup

Open a terminal and run:

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

Edit the `.env` file and add your API key:

```bash
# For Anthropic (Claude)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# OR for OpenAI (GPT)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

Start the backend server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Keep this terminal running!**

### Step 2: Frontend Setup

Open a **new terminal** window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# (Optional) Create environment file
cp .env.example .env

# Start development server
npm run dev
```

You should see:
```
  VITE v5.0.11  ready in 500 ms

  ➜  Local:   http://localhost:5173/
```

### Step 3: Open the Application

Open your browser and go to: **http://localhost:5173**

You should see the Knowledge Assist RAG interface! 🎉

## 📖 Usage Guide

### 1. Upload Documents

- Drag and drop PDF, TXT, or DOCX files into the upload area
- Wait for processing (you'll see the number of chunks created)
- Multiple files can be uploaded

### 2. Ask Questions

- Type your question in the chat input
- Press Enter or click Send
- The AI will respond based on your documents

### 3. View Sources

- Click on the sources toggle to see which document chunks were used
- Each source shows:
  - Document name
  - Page number (for PDFs)
  - Relevant text snippet

### 4. Follow-up Questions

- Continue the conversation naturally
- The system maintains context for related questions

### Example Questions to Try

After uploading a document:

1. "What are the main topics discussed in this document?"
2. "Can you summarize the key points?"
3. "What does the document say about [specific topic]?"
4. "List the important dates or numbers mentioned."
5. "What conclusions does the author draw?"

## 🔌 API Endpoints

Visit **http://localhost:8000/docs** for interactive API documentation (Swagger UI).

### Upload Endpoints

- `POST /api/v1/upload/` - Upload a single file
- `POST /api/v1/upload/batch` - Upload multiple files

### Chat Endpoints

- `POST /api/v1/chat/` - Send a question and get an answer
- `DELETE /api/v1/chat/conversation/{id}` - Clear conversation history

### Document Endpoints

- `GET /api/v1/documents/` - List uploaded documents
- `DELETE /api/v1/documents/{id}` - Delete a document

## ⚙️ Configuration

### Backend Configuration (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | AI provider (anthropic/openai) | anthropic |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `LLM_MODEL` | Model to use | claude-3-5-sonnet-20241022 |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `RETRIEVAL_K` | Number of chunks to retrieve | 4 |
| `MAX_UPLOAD_SIZE` | Max file size in bytes | 10485760 (10MB) |
| `EMBEDDING_MODEL` | Embedding model | sentence-transformers/all-MiniLM-L6-v2 |

### Frontend Configuration (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | http://localhost:8000 |

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError`
- **Solution**: Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

**Problem**: ChromaDB initialization fails
- **Solution**: Delete `backend/app/storage/chroma_db/` and restart

**Problem**: Out of memory when loading embeddings
- **Solution**: Use a smaller embedding model or reduce batch size

**Problem**: `Invalid API key`
- **Solution**: Check that you've correctly set your API key in `backend/.env`

### Frontend Issues

**Problem**: `ENOENT: no such file or directory`
- **Solution**: Make sure you're in the `frontend` directory and ran `npm install`

**Problem**: CORS errors
- **Solution**: Ensure backend `ALLOWED_ORIGINS` includes your frontend URL

**Problem**: File upload fails
- **Solution**: Check file size limits and supported file types

**Problem**: Cannot connect to backend
- **Solution**: Ensure the backend is running on port 8000

### API Errors

**Error**: `Rate limit exceeded`
- **Solution**: You've hit your API provider's rate limit. Wait a few minutes and try again.

## 🧪 Development

### Backend Development

Run tests:
```bash
cd backend
pytest
```

Format code:
```bash
black app/
```

### Frontend Development

Type checking:
```bash
cd frontend
npm run build
```

Linting:
```bash
npm run lint
```

## 🚢 Production Deployment

### Production Checklist

Before deploying to production:

- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Set up database for document metadata (PostgreSQL)
- [ ] Use production vector store (Pinecone, Weaviate)
- [ ] Add file virus scanning
- [ ] Implement user quotas
- [ ] Set up HTTPS
- [ ] Configure CDN for frontend
- [ ] Add error tracking (Sentry)
- [ ] Set up CI/CD pipeline
- [ ] Add comprehensive test suite
- [ ] Implement caching (Redis)
- [ ] Configure backup strategy

### Backend Deployment

1. Set up a production database (PostgreSQL recommended)
2. Use a production-grade vector store (Pinecone, Weaviate, or managed ChromaDB)
3. Use a reverse proxy (Nginx)
4. Set up HTTPS with SSL certificates

Example with Docker:
```bash
cd backend
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend
```

### Frontend Deployment

1. Build for production:
```bash
cd frontend
npm run build
```

2. Serve the `dist` folder with a static file server or CDN

3. Update `VITE_API_BASE_URL` to point to your production API

## 📊 What's Included

### Backend Components

✅ FastAPI application with CORS and lifecycle management
✅ Pydantic request/response models with validation
✅ File upload endpoints (single and batch)
✅ Chat endpoint with conversation support
✅ Document management endpoints
✅ VectorStoreService: ChromaDB integration
✅ DocumentProcessor: PDF, TXT, DOCX support with chunking
✅ RAGService: Complete LangChain RAG pipeline
✅ Embedding generation with Sentence Transformers
✅ Support for both Anthropic and OpenAI LLMs
✅ Comprehensive error handling

### Frontend Components

✅ FileUploader: Drag-and-drop with react-dropzone
✅ ChatWindow: Full chat interface with auto-scroll
✅ Message: Individual message display with markdown
✅ SourceCitation: Expandable source references
✅ Type-safe API client with Axios
✅ Loading states and error handling
✅ Modern, responsive CSS design
✅ Mobile-friendly layout

### Features

✅ PDF support with page numbers
✅ TXT and DOCX support
✅ File size and type validation
✅ Automatic text chunking with overlap
✅ Vector embedding generation
✅ Semantic search with ChromaDB
✅ Conversational context management
✅ Source citations with metadata
✅ Markdown response formatting
✅ Typing indicators

## ⚠️ Known Limitations

1. **Document Storage**: Files are stored locally (use S3/cloud storage for production)
2. **Vector Store**: ChromaDB is local (use managed service for scale)
3. **No Authentication**: Open access (add auth for production)
4. **No Persistence**: Conversation history is in-memory
5. **Rate Limiting**: None implemented (add for production)

## ⏱️ Estimated Setup Time

- **Backend Setup**: 5-10 minutes
- **Frontend Setup**: 5 minutes
- **First Document Upload**: 1-2 minutes (embeddings download)
- **Total Time to Running**: ~15 minutes

## 🔗 Resources

- **API Documentation**: http://localhost:8000/docs
- **Frontend Dev Server**: http://localhost:5173
- **Backend Health Check**: http://localhost:8000/health

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- UI components by [Lucide Icons](https://lucide.dev/)

## 💬 Support

For issues and questions:
- Check the troubleshooting section above
- Review the [API documentation](http://localhost:8000/docs)
- Open an issue on GitHub

## 🎯 Stopping the Application

To stop the servers:

1. **Backend**: Press `Ctrl+C` in the backend terminal
2. **Frontend**: Press `Ctrl+C` in the frontend terminal

To deactivate the Python virtual environment:
```bash
deactivate
```

---
