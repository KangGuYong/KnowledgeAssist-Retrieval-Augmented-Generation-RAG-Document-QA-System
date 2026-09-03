"""LCEL로 재작성한 RAG 체인이 LLM에게 정확히 같은 문자열을 보내는지 고정한다.

재작성에서 깨지기 쉬운 것은 "체인이 도는가"가 아니라 "프롬프트 조립이 이전과
같은가"이다. ConversationalRetrievalChain의 동작(대화 이력 접기, 첫 질문의
condense 단락, 재작성 질문 전파, 컨텍스트 조립)을 소스에서 읽어 명세로 옮긴 것이
이 파일이다. 설계 문서는 docs/superpowers/specs/2026-09-02-lcel-rewrite-design.md.

LLM은 프롬프트를 기록하는 대역이며 실제 Ollama를 부르지 않는다.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import app.services.rag_service as rag_module
from app.services.rag_service import RAGService


class FakeVectorStore:
    """Stands in for Chroma: records how it was called, returns canned results."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.results


class RecordingLLM:
    """받은 프롬프트를 종류별로 기록하고 정해진 응답을 돌려준다.

    두 호출 지점(condense/answer)이 같은 self.llm을 쓰므로, 프롬프트 꼬리로
    구분한다. condense 프롬프트는 반드시 "Standalone question:"으로 끝난다.
    """

    def __init__(self, condensed="재작성된 질문", answer="답변", usage=None):
        self.condensed = condensed
        self.answer = answer
        self.usage = usage
        self.calls = []

    def __call__(self, prompt_value):
        text = prompt_value.to_string()
        kind = "condense" if text.rstrip().endswith("Standalone question:") else "answer"
        self.calls.append((kind, text))
        content = self.condensed if kind == "condense" else self.answer
        if self.usage is None:
            return AIMessage(content=content)
        return AIMessage(content=content, usage_metadata=self.usage)

    def prompts(self, kind):
        return [text for recorded_kind, text in self.calls if recorded_kind == kind]


def build_service(monkeypatch, llm, results=(), reorder=False, llm_max_attempts=1):
    """__init__ 없이 RAGService를 세운다(임베딩 모델·Ollama 연결 회피)."""
    monkeypatch.setattr(
        rag_module, "settings",
        SimpleNamespace(
            retrieval_k=5,
            retrieval_reorder=reorder,
            ocr_block_prefix="[이미지 텍스트]",
            llm_max_attempts=llm_max_attempts,
        ),
    )
    service = RAGService.__new__(RAGService)
    service.vector_store = FakeVectorStore(list(results))
    service.llm = RunnableLambda(llm)
    service.conversation_histories = {}
    return service


def _one_chunk(text="본문"):
    return [(Document(page_content=text, metadata={}), 0.9)]


def test_first_question_skips_the_condense_llm_call(monkeypatch):
    """이력이 비면 재작성 LLM 호출 자체가 일어나지 않아야 한다.

    요청당 LLM 호출 수를 결정하는 단락이므로 동작 명세의 일부다.
    """
    llm = RecordingLLM()
    service = build_service(monkeypatch, llm, _one_chunk())

    result = asyncio.run(service.ask_question("생활SOC란?"))

    assert llm.prompts("condense") == []
    assert len(llm.prompts("answer")) == 1
    # 검색에도 원 질문이 그대로 들어간다.
    assert service.vector_store.calls[0]["query"] == "생활SOC란?"
    assert result["answer"] == "답변"


def test_follow_up_folds_the_history_into_the_condense_prompt(monkeypatch):
    """두 번째 턴에서만 condense가 1회 일어나고, 이력은 역할 접두로 접힌다."""
    llm = RecordingLLM()
    service = build_service(monkeypatch, llm, _one_chunk())

    asyncio.run(service.ask_question("생활SOC란?", conversation_id="c1"))
    asyncio.run(service.ask_question("그 예산은?", conversation_id="c1"))

    condense_prompts = llm.prompts("condense")
    assert len(condense_prompts) == 1
    # ConversationalRetrievalChain._get_chat_history가 만들던 형식 그대로.
    assert "\nHuman: 생활SOC란?\nAssistant: 답변" in condense_prompts[0]
    assert "Follow Up Input: 그 예산은?" in condense_prompts[0]


def test_the_rewritten_question_reaches_both_search_and_the_answer_prompt(monkeypatch):
    """rephrase_question=True 재현: 원 질문이 아니라 재작성된 질문이 흐른다."""
    llm = RecordingLLM(condensed="생활SOC 확충 계획의 예산은?")
    service = build_service(monkeypatch, llm, _one_chunk())

    asyncio.run(service.ask_question("생활SOC란?", conversation_id="c1"))
    asyncio.run(service.ask_question("그 예산은?", conversation_id="c1"))

    assert service.vector_store.calls[-1]["query"] == "생활SOC 확충 계획의 예산은?"

    answer_prompt = llm.prompts("answer")[-1]
    assert "생활SOC 확충 계획의 예산은?" in answer_prompt
    assert "그 예산은?" not in answer_prompt


def test_each_chunk_carries_a_source_header(monkeypatch):
    """QA_PROMPT 규칙 2번이 요구하는 인용을 실제로 할 수 있어야 한다.

    문서명이 컨텍스트에 없던 동안 규칙 2번은 지어내야만 지킬 수 있는 규칙이었다.
    헤더 형식을 규칙이 요구하는 형식과 같게 두었으므로 LLM은 그대로 옮기면 된다.
    """
    llm = RecordingLLM()
    results = [
        (Document(page_content="첫 청크", metadata={"filename": "a.pdf", "page": 3}), 0.9),
        (Document(page_content="둘째 청크", metadata={"filename": "b.pdf", "page": 7}), 0.5),
    ]
    service = build_service(monkeypatch, llm, results)

    asyncio.run(service.ask_question("질문"))

    answer_prompt = llm.prompts("answer")[0]
    assert "[출처: a.pdf, p.3]\n첫 청크" in answer_prompt
    assert "[출처: b.pdf, p.7]\n둘째 청크" in answer_prompt
    # 청크 사이는 여전히 빈 줄로 이어붙인다.
    assert "첫 청크\n\n[출처: b.pdf" in answer_prompt


def test_a_chunk_without_a_page_gets_a_header_without_one(monkeypatch):
    """페이지는 PDF에만 있다. TXT/DOCX에서 'p.None'이 나오면 안 된다."""
    llm = RecordingLLM()
    results = [(Document(page_content="본문", metadata={"filename": "메모.txt"}), 0.9)]
    service = build_service(monkeypatch, llm, results)

    asyncio.run(service.ask_question("질문"))

    answer_prompt = llm.prompts("answer")[0]
    assert "[출처: 메모.txt]\n본문" in answer_prompt
    assert "p.None" not in answer_prompt


def test_history_accumulates_in_order_and_clears(monkeypatch):
    llm = RecordingLLM()
    service = build_service(monkeypatch, llm, _one_chunk())

    asyncio.run(service.ask_question("질문1", conversation_id="c1"))

    history = service.conversation_histories["c1"]
    assert [(m.type, m.content) for m in history] == [
        ("human", "질문1"),
        ("ai", "답변"),
    ]

    service.clear_conversation("c1")
    assert "c1" not in service.conversation_histories


def test_conversations_do_not_leak_into_each_other(monkeypatch):
    """다른 conversation_id는 서로의 이력을 보지 못한다.

    두 번째 대화의 첫 질문에서 condense가 일어나면 이력이 새고 있는 것이다.
    """
    llm = RecordingLLM()
    service = build_service(monkeypatch, llm, _one_chunk())

    asyncio.run(service.ask_question("A질문", conversation_id="a"))
    asyncio.run(service.ask_question("B질문", conversation_id="b"))

    assert llm.prompts("condense") == []
    assert [m.content for m in service.conversation_histories["a"]] == ["A질문", "답변"]
    assert [m.content for m in service.conversation_histories["b"]] == ["B질문", "답변"]


def test_a_failed_turn_is_not_written_to_history(monkeypatch):
    """체인이 memory를 저장하던 시점과 같아야 한다: 실패한 턴은 남지 않는다."""
    def boom(_prompt_value):
        raise RuntimeError("ollama down")

    service = build_service(monkeypatch, boom, _one_chunk())

    with pytest.raises(RuntimeError):
        asyncio.run(service.ask_question("질문", conversation_id="c1"))

    assert service.conversation_histories["c1"] == []


@pytest.mark.parametrize(
    "chunk_text, notice_expected",
    [
        ("표 안의 값은 [이미지 텍스트] 3,200억 원", True),
        ("도표가 섞이지 않은 평범한 본문", False),
    ],
)
def test_ocr_notice_rides_along_only_when_the_context_has_the_marker(
    monkeypatch, chunk_text, notice_expected
):
    """OCR 주의문은 컨텍스트에 마커가 있을 때만 붙는다.

    항상 붙이면 도표와 무관한 답변에서도 "도표를 직접 확인하라"는 헤지가 따라붙고,
    매 요청 토큰을 낭비한다. 양쪽 방향을 모두 검사해야 상수로 박아넣은 코드를
    잡을 수 있다.
    """
    llm = RecordingLLM()
    service = build_service(monkeypatch, llm, _one_chunk(chunk_text))

    asyncio.run(service.ask_question("질문"))

    answer_prompt = llm.prompts("answer")[0]
    assert ("문자 인식(OCR)으로" in answer_prompt) is notice_expected


class FlakyLLM:
    """정해진 횟수만큼 실패한 뒤 성공하는 LLM 대역."""

    def __init__(self, error, failures):
        self.error = error
        self.failures = failures
        self.attempts = 0

    def __call__(self, _prompt_value):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return AIMessage(content="답변")


@pytest.mark.parametrize(
    "error",
    [ConnectionError("ollama unreachable"), httpx.ReadTimeout("too slow")],
)
def test_a_transient_llm_failure_is_retried(monkeypatch, error):
    """원격 Ollama가 잠깐 끊긴 것만으로 대화가 500이 되면 안 된다."""
    # 한 번만 실패시킨다. 재시도가 일어난다는 사실을 보여주는 데는 충분하고,
    # with_retry의 지수 백오프가 테스트 실행 시간을 잡아먹지 않는다.
    llm = FlakyLLM(error, failures=1)
    service = build_service(monkeypatch, llm, _one_chunk(), llm_max_attempts=3)

    result = asyncio.run(service.ask_question("질문"))

    assert llm.attempts == 2
    assert result["answer"] == "답변"


def test_a_permanent_llm_failure_is_not_retried(monkeypatch):
    """모델 없음·잘못된 요청처럼 다시 걸어도 같은 실패는 재시도하지 않는다."""
    llm = FlakyLLM(ValueError("model not found"), failures=99)
    service = build_service(monkeypatch, llm, _one_chunk(), llm_max_attempts=3)

    with pytest.raises(ValueError):
        asyncio.run(service.ask_question("질문"))

    assert llm.attempts == 1


def test_retry_can_be_turned_off(monkeypatch):
    """llm_max_attempts=1이면 일시적 실패도 그대로 올라온다."""
    llm = FlakyLLM(ConnectionError("ollama unreachable"), failures=99)
    service = build_service(monkeypatch, llm, _one_chunk(), llm_max_attempts=1)

    with pytest.raises(ConnectionError):
        asyncio.run(service.ask_question("질문"))

    assert llm.attempts == 1


def _usage(inp, out):
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def test_token_usage_covers_only_the_answer_call_on_a_first_question(monkeypatch):
    """첫 질문은 LLM을 한 번만 부르므로 그 한 번의 사용량만 나와야 한다."""
    llm = RecordingLLM(usage=_usage(100, 20))
    service = build_service(monkeypatch, llm, _one_chunk())

    result = asyncio.run(service.ask_question("질문"))

    assert llm.prompts("condense") == []
    assert result["token_usage"] == _usage(100, 20)


def test_token_usage_sums_both_llm_calls_on_a_follow_up(monkeypatch):
    """후속 질문은 재작성 + 답변 두 번을 부르므로 합계여야 한다.

    답변 호출분만 세면 재작성 비용이 조용히 빠진다.
    """
    llm = RecordingLLM(usage=_usage(100, 20))
    service = build_service(monkeypatch, llm, _one_chunk())

    asyncio.run(service.ask_question("첫 질문", conversation_id="c1"))
    result = asyncio.run(service.ask_question("후속 질문", conversation_id="c1"))

    assert len(llm.prompts("condense")) == 1
    assert result["token_usage"] == _usage(200, 40)


def test_missing_usage_metadata_reports_zeros(monkeypatch):
    """사용량을 주지 않는 공급자에서도 답변은 정상이어야 한다."""
    llm = RecordingLLM(usage=None)
    service = build_service(monkeypatch, llm, _one_chunk())

    result = asyncio.run(service.ask_question("질문"))

    assert result["answer"] == "답변"
    assert result["token_usage"] == _usage(0, 0)
