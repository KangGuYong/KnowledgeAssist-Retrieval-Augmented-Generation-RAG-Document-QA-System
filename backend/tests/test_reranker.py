"""크로스인코더 리랭커. 실제 모델은 로드하지 않고 대역으로 검증한다.

이 모듈이 지켜야 할 것은 두 가지다. (질문, 청크 본문) 쌍을 만들어 모델에
넘기는 것, 그리고 모델을 첫 사용 시점까지 만들지 않는 것.
"""

from langchain_core.documents import Document

from app.services.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """predict()에 들어온 쌍을 기록하고 정해진 점수를 돌려준다."""

    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores


def _reranker_with(model):
    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    reranker._model = model
    return reranker


def test_score_pairs_the_query_with_each_chunk():
    model = FakeCrossEncoder([0.9, 0.1])
    reranker = _reranker_with(model)
    docs = [Document(page_content="첫 청크"), Document(page_content="둘째 청크")]

    scores = reranker.score("생활SOC란?", docs)

    assert model.pairs == [("생활SOC란?", "첫 청크"), ("생활SOC란?", "둘째 청크")]
    assert scores == [0.9, 0.1]


def test_empty_document_list_does_not_touch_the_model():
    model = FakeCrossEncoder([])
    reranker = _reranker_with(model)

    assert reranker.score("질문", []) == []
    assert model.pairs is None


def test_model_is_not_built_until_first_use():
    """2.27GB 모델을 임포트만으로 올리면 안 된다."""
    reranker = CrossEncoderReranker(model_name="fake", device="cpu")

    assert reranker._model is None
