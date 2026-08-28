from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SourceDocument(BaseModel):
    """Source document chunk with metadata."""
    content: str = Field(..., description="Text content of the chunk")
    document_name: str = Field(..., description="Original document filename")
    document_id: str = Field(..., description="Document ID")
    page: Optional[int] = Field(None, description="Page number (for PDFs)")
    chunk_index: int = Field(..., description="Chunk index in document")
    similarity_score: Optional[float] = Field(None, description="Relevance score")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] = Field(default_factory=list, description="Source citations")
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Unique message ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The document discusses machine learning fundamentals...",
                "sources": [
                    {
                        "content": "Machine learning is a subset of AI...",
                        "document_name": "ml_intro.pdf",
                        "document_id": "doc_123",
                        "page": 5,
                        "chunk_index": 12,
                        "similarity_score": 0.89
                    }
                ],
                "conversation_id": "conv_123",
                "message_id": "msg_456",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class UploadResponse(BaseModel):
    """Response for file upload."""
    document_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    num_chunks: int = Field(..., description="Number of chunks created")
    status: str = Field(default="processed", description="Processing status")
    message: str = Field(..., description="Status message")


class DocumentInfo(BaseModel):
    """Document metadata."""
    document_id: str
    filename: str
    upload_date: datetime
    num_chunks: int
    file_size: int
