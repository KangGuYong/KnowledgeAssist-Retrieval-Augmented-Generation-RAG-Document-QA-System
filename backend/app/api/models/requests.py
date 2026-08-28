from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    document_ids: Optional[list[str]] = Field(None, description="Specific documents to query")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main topics discussed in the document?",
                "conversation_id": "conv_123",
                "document_ids": ["doc_abc", "doc_xyz"]
            }
        }


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    name: Optional[str] = Field(None, description="Optional conversation name")
