from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_transformers import LongContextReorder
from typing import Any, Dict, List, Optional
import logging
import uuid

from app.config import get_settings
from app.services.vector_store import get_vector_store
from app.api.models.responses import SourceDocument

logger = logging.getLogger(__name__)
settings = get_settings()

# LLM은 긴 컨텍스트의 가운데를 잘 놓친다(Liu et al. 2023, "Lost in the Middle").
# 관련성 높은 청크를 양 끝으로 보내 이 취약 구간을 피한다. 상태가 없는 객체라
# 요청마다 새로 만들 이유가 없다.
_LONG_CONTEXT_REORDER = LongContextReorder()

# 문맥 중 '[이미지 텍스트]'는 OCR로 추출된 것이라 오탈자가 있을 수 있다. 청크
# 본문에 이미 인라인으로 박혀 있으므로(mineru_client._format_ocr_block), 여기서는 LLM에게
# 그 마커를 어떻게 다뤄야 하는지만 알려준다.
QA_PROMPT = PromptTemplate(
    template="""당신은 주어진 컨텍스트를 기반으로 정확한 정보를 제공하는 AI 어시스턴트입니다.

## 검색된 문서
{context}

문맥 중 '[이미지 텍스트]'로 표시된 부분은 문서의 도표·이미지에서 문자 인식(OCR)으로
추출한 것이라 오탈자가 있을 수 있다. 이를 근거로 답할 때는 인명·지명 등 고유명사를
단정하지 말고, 해당 페이지의 도표를 직접 확인하도록 안내하라.

## 응답 규칙
1. **정확성**: 컨텍스트에 명시된 정보만 사용하세요.
2. **출처 인용**: 답변에 사용한 정보의 출처를 [출처: 문서명] 형식으로 표시하세요.
3. **불확실성 표현**: 정보가 불완전하면 "~로 추정됩니다" 등으로 표현하세요.
4. **없는 정보**: 컨텍스트에 없는 정보는 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.

## 사용자 질문
{question}

위 규칙을 따라 답변하세요:""",
    input_variables=["context", "question"],
)


class ScoringRetriever(BaseRetriever):
    """검색 결과에 유사도 점수를 메타데이터로 실어 보낸다.

    vector_store.as_retriever()가 반환하는 기본 리트리버는 문서만 돌려주고
    점수를 버린다. similarity_search_with_relevance_scores를 직접 호출해
    doc.metadata['similarity_score']에 채워, 답변 생성과 동일한 검색 결과에서
    나온 점수를 그대로 출처 응답까지 이어지게 한다.

    reorder=True면 관련성 높은 청크를 컨텍스트 양 끝으로 재배치한다. 점수는
    메타데이터에 남으므로, 사용자에게 보여줄 출처 목록은 _format_sources가
    다시 점수 순으로 되돌린다.
    """

    vector_store: Any
    k: int
    search_filter: Optional[dict] = None
    reorder: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k, filter=self.search_filter
        )
        # 점수는 반드시 재배치 "전"에 심는다. 재배치하면 문서 순서가 바뀌면서
        # results의 (doc, score) 짝을 더는 위치로 복원할 수 없다.
        for doc, score in results:
            doc.metadata["similarity_score"] = score

        docs = [doc for doc, _ in results]
        if not self.reorder:
            return docs
        # transform_documents는 Sequence를 돌려주므로 List로 맞춘다.
        return list(_LONG_CONTEXT_REORDER.transform_documents(docs))


class RAGService:
    """Service for RAG-powered question answering."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = self._initialize_llm()
        # Store conversation memories by conversation_id
        self.conversation_memories: Dict[str, ConversationBufferMemory] = {}

    def _initialize_llm(self):
        """Initialize the LLM based on provider setting."""
        if settings.llm_provider != "ollama":
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    def _get_or_create_memory(self, conversation_id: str) -> ConversationBufferMemory:
        """Get existing conversation memory or create new one."""
        if conversation_id not in self.conversation_memories:
            self.conversation_memories[conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.conversation_memories[conversation_id]

    async def ask_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None
    ) -> dict:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            conversation_id: Optional conversation ID for context
            document_ids: Optional list of specific document IDs to search

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        # Get or create conversation memory
        memory = self._get_or_create_memory(conversation_id)

        # Set up retriever with optional document filtering
        search_kwargs = {"k": settings.retrieval_k}
        if document_ids:
            search_kwargs["filter"] = {"document_id": {"$in": document_ids}}

        retriever = ScoringRetriever(
            vector_store=self.vector_store,
            k=search_kwargs["k"],
            search_filter=search_kwargs.get("filter"),
            reorder=settings.retrieval_reorder,
        )

        # Create conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=True
        )

        # Get response
        try:
            result = qa_chain({"question": question})

            # Format source documents
            sources = self._format_sources(result.get("source_documents", []))

            return {
                "answer": result["answer"],
                "sources": sources,
                "conversation_id": conversation_id,
                "message_id": f"msg_{uuid.uuid4().hex[:12]}"
            }

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    def _format_sources(self, source_docs: list) -> list[SourceDocument]:
        """Format source documents for response.

        리트리버가 컨텍스트 배치용으로 순서를 바꿨더라도(ScoringRetriever.reorder)
        사용자에게 보여줄 출처는 관련성이 높은 순서여야 하므로 점수로 되돌린다.
        """
        formatted_sources = []

        # 점수가 없는 문서(리트리버를 거치지 않은 경우)는 0.0으로 취급해 맨 뒤로.
        ordered_docs = sorted(
            source_docs,
            key=lambda doc: doc.metadata.get("similarity_score") or 0.0,
            reverse=True,
        )

        for doc in ordered_docs:
            document_id = doc.metadata.get("document_id", "")
            # Chroma metadata can only hold scalars, so image_ids is stored as a
            # comma-joined string (document_processor.load_pdf); split it back out.
            image_ids_raw = doc.metadata.get("image_ids") or ""
            image_ids = image_ids_raw.split(",") if image_ids_raw else []
            image_urls = (
                [f"/api/v1/documents/{document_id}/images/{image_id}" for image_id in image_ids]
                if document_id
                else []
            )

            source = SourceDocument(
                content=doc.page_content,
                document_name=doc.metadata.get("filename", "Unknown"),
                document_id=document_id,
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                similarity_score=doc.metadata.get("similarity_score"),
                image_urls=image_urls,
            )
            formatted_sources.append(source)

        return formatted_sources

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history."""
        if conversation_id in self.conversation_memories:
            del self.conversation_memories[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
