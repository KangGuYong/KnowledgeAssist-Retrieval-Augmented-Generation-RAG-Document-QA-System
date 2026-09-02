"""langchain_community의 LongContextReorder 동작을 계약으로 고정한다.

재배치 알고리즘을 직접 소유하지 않고 라이브러리에 의존하기로 했으므로
(docs/superpowers/specs/2026-09-02-search-result-reordering-design.md 2절),
버전을 올렸을 때 배치가 달라지면 이 테스트가 깨져서 알려줘야 한다.
"""

import pytest
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder


def docs_numbered(n):
    """관련성 내림차순 문서 n개. '1'이 가장 관련성 높다."""
    return [Document(page_content=str(i + 1), metadata={}) for i in range(n)]


def reorder(docs):
    return [d.page_content for d in LongContextReorder().transform_documents(docs)]


@pytest.mark.parametrize(
    "n, expected",
    [
        # 홀수: 1등이 맨 앞
        (5, ["1", "3", "5", "4", "2"]),
        (7, ["1", "3", "5", "7", "6", "4", "2"]),
        # 짝수: 방향이 뒤집혀 1등이 맨 뒤 (설계 문서 3.6절)
        (4, ["2", "4", "3", "1"]),
        (10, ["2", "4", "6", "8", "10", "9", "7", "5", "3", "1"]),
    ],
)
def test_reordering_is_stable_across_library_versions(n, expected):
    assert reorder(docs_numbered(n)) == expected


def test_most_relevant_documents_end_up_at_the_edges():
    """홀짝과 무관하게 성립해야 하는 본질적 성질.

    가장자리에서 안쪽으로 짝지어 보면 관련성 순서가 유지된다.
    예: 10개면 양 끝이 {1,2}, 그 안쪽이 {3,4}, ... 가운데가 {9,10}.
    """
    for n in (4, 5, 7, 10):
        result = [int(x) for x in reorder(docs_numbered(n))]
        for i in range(n // 2):
            pair = {result[i], result[n - 1 - i]}
            expected_pair = {2 * i + 1, 2 * i + 2}
            assert pair == expected_pair, f"n={n}, 바깥에서 {i}번째 쌍이 {pair}"


def test_empty_and_single_document_pass_through():
    assert reorder([]) == []
    assert reorder(docs_numbered(1)) == ["1"]


def test_input_list_is_not_mutated():
    """_litm_reordering이 내부에서 reverse()를 호출하므로 확인해 둔다."""
    docs = docs_numbered(5)
    LongContextReorder().transform_documents(docs)

    assert [d.page_content for d in docs] == ["1", "2", "3", "4", "5"]
