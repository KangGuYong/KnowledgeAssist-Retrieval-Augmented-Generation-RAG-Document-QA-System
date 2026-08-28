from fastapi import APIRouter, HTTPException, status
import logging

from app.api.models.requests import ChatRequest
from app.api.models.responses import ChatResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for asking questions about uploaded documents.

    This endpoint:
    1. Receives a question and optional conversation context
    2. Retrieves relevant document chunks from vector store
    3. Generates an answer using the LLM with retrieved context
    4. Returns the answer with source citations
    """
    try:
        result = await rag_service.ask_question(
            question=request.question,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history."""
    try:
        rag_service.clear_conversation(conversation_id)
        return {"message": f"Conversation {conversation_id} cleared"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
