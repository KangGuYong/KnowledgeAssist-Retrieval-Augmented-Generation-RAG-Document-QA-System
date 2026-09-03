from langchain_community.document_transformers import LongContextReorder
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from typing import Any, Dict, List, Optional
import httpx
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

# 컨텍스트는 청크마다 출처 헤더를 한 줄 붙이고 빈 줄로 이어붙인 것이다.
# 헤더 형식은 QA_PROMPT 규칙 2번이 요구하는 인용 형식과 일부러 같게 두었다.
# 그래야 LLM이 지어내지 않고 눈앞의 문자열을 그대로 옮길 수 있다.
_DOCUMENT_SEPARATOR = "\n\n"

# ConversationalRetrievalChain._get_chat_history가 쓰던 역할 접두다.
_ROLE_PREFIXES = {"human": "Human: ", "ai": "Assistant: "}

_USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")
_NO_USAGE = {key: 0 for key in _USAGE_KEYS}

# 체인을 AIMessage까지만 세우고 본문 추출은 여기서 한다. 그래야 같은 응답에서
# usage_metadata도 꺼낼 수 있다 - 파서가 체인 끝에 붙으면 메시지가 버려진다.
_TEXT_PARSER = StrOutputParser()

# ollama 클라이언트는 httpx.ConnectError를 내장 ConnectionError로 바꿔 던지고,
# 타임아웃은 httpx 예외 그대로 올려보낸다. 둘 다 다시 걸면 성공할 수 있는 실패다.
# ollama.ResponseError는 404(모델 없음)와 500이 섞여 있어 재시도 대상에서 뺀다.
_TRANSIENT_LLM_ERRORS = (ConnectionError, httpx.TimeoutException)

# 문맥 중 '[이미지 텍스트]'는 OCR로 추출된 것이라 오탈자가 있을 수 있다. 청크
# 본문에 이미 인라인으로 박혀 있으므로(mineru_client._format_ocr_block), 여기서는 LLM에게
# 그 마커를 어떻게 다뤄야 하는지만 알려준다.
#
# 검색된 청크에 마커가 하나도 없으면 이 문단은 붙이지 않는다(_ocr_notice_for).
# 앞뒤 개행까지 포함해야 붙일 때와 뗄 때 모두 문단 간격이 맞는다.
_OCR_NOTICE = """
문맥 중 '[이미지 텍스트]'로 표시된 부분은 문서의 도표·이미지에서 문자 인식(OCR)으로
추출한 것이라 오탈자가 있을 수 있다. 이를 근거로 답할 때는 인명·지명 등 고유명사를
단정하지 말고, 해당 페이지의 도표를 직접 확인하도록 안내하라.
"""

QA_PROMPT = PromptTemplate(
    template="""당신은 주어진 컨텍스트를 기반으로 정확한 정보를 제공하는 AI 어시스턴트입니다.

## 검색된 문서
{context}
{ocr_notice}
## 응답 규칙
1. **정확성**: 컨텍스트에 명시된 정보만 사용하세요.
2. **출처 인용**: 답변에 사용한 정보의 출처를 [출처: 문서명] 형식으로 표시하세요.
3. **불확실성 표현**: 정보가 불완전하면 "~로 추정됩니다" 등으로 표현하세요.
4. **없는 정보**: 컨텍스트에 없는 정보는 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.
5. **검색된 문서 없음**: 검색된 문서가 없다면 사용자 질문에 성실하게 답을 하세요.

## 사용자 질문
{question}

위 규칙을 따라 답변하세요:""",
    input_variables=["context", "ocr_notice", "question"],
)

# ConversationalRetrievalChain이 쓰던 기본 프롬프트를 그대로 옮겨 왔다
# (langchain_classic/chains/conversational_retrieval/prompts.py). 영어 지시문이지만
# "in its original language" 덕분에 한국어 질문은 한국어로 재작성된다. 검토된 적 없는
# 서드파티 기본값이었고, 이제는 우리 것이라 바꾸려면 커밋이 필요하다.
CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)


def _format_chat_history(messages: List[BaseMessage]) -> str:
    """대화 이력을 condense 프롬프트에 넣을 한 덩어리 문자열로 접는다.

    ConversationalRetrievalChain._get_chat_history의 동작을 그대로 옮겼다.
    턴마다 앞에 개행과 역할 접두를 붙이고, 내용이 빈 메시지는 건너뛴다.
    첫 턴 앞의 개행까지 같아야 프롬프트가 이전과 동일해진다.
    """
    buffer = ""
    for message in messages:
        if not message.content:
            continue
        prefix = _ROLE_PREFIXES.get(message.type, f"{message.type}: ")
        buffer += f"\n{prefix}{message.content}"
    return buffer


def _format_document(doc: Document) -> str:
    """청크 하나를 출처 헤더 + 본문으로 만든다.

    페이지는 PDF에만 있으므로(TXT/DOCX는 None) 있을 때만 붙인다. 문서명이
    없는 경우의 "Unknown"은 _format_sources가 쓰는 값과 맞춘 것이다.
    """
    filename = doc.metadata.get("filename", "Unknown")
    page = doc.metadata.get("page")
    header = f"[출처: {filename}, p.{page}]" if page is not None else f"[출처: {filename}]"
    return f"{header}\n{doc.page_content}"


def _format_context(docs: List[Document]) -> str:
    """검색된 청크를 컨텍스트 문자열로 조립한다."""
    return _DOCUMENT_SEPARATOR.join(_format_document(doc) for doc in docs)


def _usage_of(message: BaseMessage) -> Dict[str, int]:
    """AIMessage에서 토큰 사용량을 꺼낸다.

    usage_metadata는 공급자가 채워주는 값이라 없을 수 있다. 그때는 0으로
    둔다 - 사용량을 못 셌다는 이유로 답변이 실패하면 안 된다.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    return {key: usage.get(key, 0) or 0 for key in _USAGE_KEYS}


def _add_usage(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """한 번의 질문이 LLM을 두 번 부르므로(재작성 + 답변) 합산한다."""
    return {key: left[key] + right[key] for key in _USAGE_KEYS}


def _ocr_notice_for(context: str) -> str:
    """OCR 마커가 실제로 컨텍스트에 있을 때만 주의문을 돌려준다.

    항상 붙이면 도표가 섞이지 않은 답변에서도 LLM이 "도표를 직접 확인하라"고
    불필요하게 헤지하고, 매 요청 토큰도 낭비한다.
    """
    return _OCR_NOTICE if settings.ocr_block_prefix in context else ""


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
        # conversation_id -> 그 대화의 메시지 목록. 프로세스가 재시작되면 전부
        # 사라지고, 워커를 여러 개 띄우면 공유되지 않는다.
        self.conversation_histories: Dict[str, List[BaseMessage]] = {}

    def _initialize_llm(self):
        """Initialize the LLM based on provider setting."""
        if settings.llm_provider != "ollama":
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            # Ollama 쪽 이름은 num_predict다. 연결하지 않던 동안 답변 길이
            # 상한이 사실상 없었다.
            num_predict=settings.max_tokens,
        )

    def _llm_with_retry(self):
        """일시적 실패에만 재시도를 붙인 LLM Runnable.

        원격 Ollama(ollama_base_url)가 순간적으로 불통이면 대화가 그대로
        500으로 끝나던 것을 막는다. 재시도 간격은 with_retry의 지수 백오프를
        따른다.
        """
        return self.llm.with_retry(
            retry_if_exception_type=_TRANSIENT_LLM_ERRORS,
            stop_after_attempt=settings.llm_max_attempts,
        )

    async def _condense_question(
        self, question: str, history: List[BaseMessage]
    ) -> tuple[str, Dict[str, int]]:
        """후속 질문을 대화 맥락 없이도 검색 가능한 독립 질문으로 바꾼다.

        첫 질문이면 이력이 비어 있으므로 LLM을 부르지 않고 원 질문을 그대로
        돌려준다. 이 단락(short-circuit)은 요청당 LLM 호출 수를 결정하므로
        동작 명세의 일부다.
        """
        chat_history = _format_chat_history(history)
        if not chat_history:
            # LLM을 부르지 않았으므로 이 턴의 사용량에 더할 것도 없다.
            return question, _NO_USAGE

        chain = CONDENSE_PROMPT | self._llm_with_retry()
        message = await chain.ainvoke(
            {"chat_history": chat_history, "question": question}
        )
        return _TEXT_PARSER.invoke(message), _usage_of(message)

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

        history = self.conversation_histories.setdefault(conversation_id, [])

        # Set up retriever with optional document filtering
        search_filter = {"document_id": {"$in": document_ids}} if document_ids else None
        retriever = ScoringRetriever(
            vector_store=self.vector_store,
            k=settings.retrieval_k,
            search_filter=search_filter,
            reorder=settings.retrieval_reorder,
        )

        try:
            standalone_question, usage = await self._condense_question(
                question, history
            )
            if standalone_question != question:
                logger.debug("Condensed question: %s", standalone_question)

            # 검색도 답변도 재작성된 질문으로 한다. QA_PROMPT의 {question}에
            # 원 질문이 아니라 이 질문이 들어가는 것은
            # ConversationalRetrievalChain의 rephrase_question=True 기본값과 같다.
            docs = await retriever.ainvoke(standalone_question)

            context = _format_context(docs)
            answer_chain = QA_PROMPT | self._llm_with_retry()
            message = await answer_chain.ainvoke(
                {
                    "context": context,
                    "ocr_notice": _ocr_notice_for(context),
                    "question": standalone_question,
                }
            )
            answer = _TEXT_PARSER.invoke(message)
            usage = _add_usage(usage, _usage_of(message))

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

        # 실패한 턴은 이력에 남기지 않는다. 체인이 메모리를 저장하던 시점과 같다.
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": self._format_sources(docs),
            "conversation_id": conversation_id,
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "token_usage": usage,
        }

    def _format_sources(self, source_docs: list) -> list[SourceDocument]:
        """Format source documents for response.

        리트리버가 컨텍스트 배치용으로 순서를 바꿨더라도(ScoringRetriever.reorder)
        사용자에게 보여줄 출처는 관련성이 높은 순서여야 하므로 점수로 되돌린다.
        """
        formatted_sources = []

        # 점수가 없는 문서(리트리버를 거치지 않은 경우)는 맨 뒤로 보낸다.
        # 0.0이 아니라 -inf인 이유: 관련성 점수는 음수가 될 수 있다. Chroma
        # 컬렉션이 hnsw:space 없이 만들어져 LangChain이 유클리드 변환식
        # (1 - distance/sqrt(2))을 쓰기 때문이다.
        ordered_docs = sorted(
            source_docs,
            key=lambda doc: doc.metadata.get("similarity_score") or float("-inf"),
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
        if conversation_id in self.conversation_histories:
            del self.conversation_histories[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
