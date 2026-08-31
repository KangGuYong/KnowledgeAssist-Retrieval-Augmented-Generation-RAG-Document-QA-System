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
    image_urls: list[str] = Field(
        default_factory=list,
        description="이 청크가 속한 페이지에서 발견된 도표 이미지 URL 목록"
    )


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
    chunking_strategy: str = Field(default="default", description="Chunking strategy used: default | semantic")


class DocumentInfo(BaseModel):
    """Document metadata."""
    document_id: str
    filename: str
    upload_date: datetime
    num_chunks: int
    file_size: int


class ParsedBlock(BaseModel):
    """MinerU content_list 블록을 원본 그대로 옮긴 것."""
    type: str
    text: Optional[str] = None
    table_body: Optional[str] = None
    image_id: Optional[str] = None


class ParsedPage(BaseModel):
    page_number: int
    blocks: list[ParsedBlock]


class ParsedDocumentSummary(BaseModel):
    """파싱 결과 목록 화면에 쓰이는 요약."""
    document_id: str
    filename: str
    created_at: str
    page_count: int


class ParsedDocumentDetail(ParsedDocumentSummary):
    """파싱 결과 상세 - 페이지별 원본 블록 전체."""
    pages: list[ParsedPage]
