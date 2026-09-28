"""크로스인코더 리랭커.

벡터 검색은 질문과 청크를 각각 따로 벡터화해 거리를 재므로 둘을 함께 읽지
않는다. 크로스인코더는 (질문, 청크) 쌍을 한 번에 입력받아 관련성을 직접
채점하므로, 벡터 검색이 뒤로 밀어낸 청크를 끌어올릴 수 있다.
"""

from functools import lru_cache
from typing import List, Optional
import logging
import threading

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """(질문, 청크) 쌍을 0~1로 채점한다.

    sentence-transformers의 CrossEncoder는 config.num_labels가 1이면
    predict()에 nn.Sigmoid()를 자동 적용한다(CrossEncoder.py:141). 기본 모델인
    dragonkue/bge-reranker-v2-m3-ko는 id2label이 하나라 여기 해당하므로 점수가
    0~1로 나오고, 화면의 퍼센트 표시에 그대로 쓸 수 있다. num_labels가 1이 아닌
    모델로 바꾸면 점수 범위가 달라져 그 표시가 어긋난다.
    """

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None
    ):
        self.model_name = model_name or settings.rerank_model #리랭커 모델명
        self.device = device or settings.rerank_device #리랭커 CPU or GPU 사용 선택
        self._model = None #동작 모댈
        self._lock = threading.Lock()

    def _build_model(self):
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker: %s (device=%s)", self.model_name, self.device
        )
        return CrossEncoder(self.model_name, device=self.device) #CrossEncoder 함수를 사용하면 허킹페이스에서 모델을 자동으로 내려받는다.

    @property
    def model(self):
        """모델은 첫 채점 때 만든다. 2.27GB라 임포트 시점에 올릴 수 없다."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._build_model()
        return self._model

    #호출시 모델 초기화: 1회 재실행시 캐시값을 사용
    def score(self, query: str, docs: List[Document]) -> List[float]:
        """docs와 같은 길이의 점수 목록을 돌려준다."""
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        return [float(score) for score in self.model.predict(pairs)]


@lru_cache()
def get_reranker() -> CrossEncoderReranker:
    """Cached reranker so the model is only loaded once per worker."""
    return CrossEncoderReranker()
