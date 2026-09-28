# 크로스인코더 리랭킹 (적극적 필터링)

- 작성일: 2026-09-03
- 상태: 설계 검토 대기
- 관련 코드: `rag_service.py`, `config.py`, `responses.py`, `SourceCitation.tsx`
- 선행 작업: [검색 결과 재배치](2026-09-02-search-result-reordering-design.md)

## 1. 문제

[검색 결과 재배치](2026-09-02-search-result-reordering-design.md)는 검색 결과의
**순서**만 바꿨다. 그 문서의 §4는 이 작업을 비대상으로 명시하며 이유를 남겼다.

> **크로스인코더 리랭킹.** k를 20~30으로 늘려 뽑고 리랭커 모델로 재채점해 상위 N개만
> 남기는 방식. 검색 품질 개선폭은 크지만 모델 로딩·GPU 메모리·질의당 지연이
> 추가되며, 이는 "무엇을 고를지"를 바꾸는 별개의 작업이다.

이 문서가 그 별개의 작업이다.

현재 `ScoringRetriever`는 Chroma에서 `retrieval_k`개(`.env` 기준 5개)를 뽑아
그대로 LLM에 넘긴다. **무엇을 고를지는 벡터 유사도가 단독으로 결정한다.**
임베딩은 질문과 청크를 각각 독립적으로 벡터화한 뒤 거리를 재므로, 질문과 청크를
함께 읽고 판단하지 않는다. 5위 밖으로 밀린 청크는 구제할 방법이 없다.

크로스인코더는 (질문, 청크) 쌍을 한 번에 입력받아 관련성을 직접 채점한다.
후보를 넓게 뽑아 재채점하면 벡터 검색이 놓친 청크를 상위로 끌어올릴 수 있다.

## 2. 설계 원칙

- **끄면 지금과 완전히 같아야 한다.** `rerank_enabled=False`에서 기존 테스트가
  수정 없이 통과하는 것이 이 작업의 안전망이다.
- **리랭킹은 품질 개선이지 정확성 요건이 아니다.** 리랭커가 죽어도 답변은 나와야
  한다. MinerU가 죽어도 pypdf로 업로드가 계속되는 것과 같은 원칙이다.
- **순서 의존성을 한곳에 모은다.** 검색·점수 부착·리랭킹·재배치를 한 메서드에
  둔다. 재배치 설계가 같은 이유로 택한 배치이며, 리랭킹도 순서에 민감하다.
- **모델 수명은 분리한다.** 무거운 모델은 자체 모듈에서 `@lru_cache` 게터로
  소유한다. `vector_store.get_embeddings()`, `ocr_service`와 같은 형태다.
- **기존 설정의 의미를 바꾸지 않는다.** `retrieval_k`는 지금도 앞으로도
  "LLM에 넣는 최종 개수"다.

## 3. 설계

### 3.1 데이터 흐름

```
k = rerank_candidate_k if rerank_enabled else retrieval_k
   │
   ▼
similarity_search_with_relevance_scores(query, k, filter)
   │  [(doc, score)] × k
   ▼
doc.metadata["similarity_score"] = score          ← 기존 그대로. 반드시 재배치 전
   │
   ├── rerank_enabled=False ─→ 그대로 통과            ← 지금과 동일한 동작
   │
   └── rerank_enabled=True
          reranker.score(query, docs)              → 0~1
          doc.metadata["rerank_score"] = s         ← 이것도 재배치 전
          점수 내림차순 정렬 → 상위 retrieval_k개
   ▼
LongContextReorder.transform_documents(docs)       ← retrieval_reorder=True일 때
   │
   ├──► 체인 컨텍스트 (LLM이 보는 순서)
   └──► _format_sources → rerank_score 내림차순(없으면 similarity_score)
```

**끈 상태에서는 Chroma에 넘기는 `k`부터 달라진다.** 30개를 뽑아 5개로 잘라내는
것이 아니라 처음부터 5개만 뽑는다. 그래야 검색 비용까지 현재와 같아진다.

`_format_sources`의 정렬 기준은 **요청 단위로 하나만 쓴다.** 한 응답의 문서는
전부 리랭킹을 거쳤거나 전부 거치지 않았거나 둘 중 하나이므로, 두 점수 체계가
한 정렬에 섞이지 않는다. 폴백했을 때도 그 요청의 모든 문서에 `rerank_score`가
없다.

### 3.2 리랭커와 재배치는 충돌하지 않는다

역할이 다르다. **리랭커는 어떤 청크를 고를지와 그 관련성 순서를 정하고,
`LongContextReorder`는 고른 것을 컨텍스트 어디에 놓을지를 정한다.**

`LongContextReorder`는 입력이 이미 관련성 내림차순이라고 가정하고 U자 배치를
만든다. 리랭킹이 그 가정을 더 잘 만족시키므로, 두 단계는 서로를 방해하지 않고
리랭킹이 재배치의 전제를 강화한다.

기존 불변식은 그대로 유지된다. **점수는 반드시 재배치 전에 심는다.** 재배치하면
문서 순서가 바뀌어 `(doc, score)` 짝을 위치로 복원할 수 없기 때문이며,
`rerank_score`에도 똑같이 적용된다.

### 3.3 리랭커 모듈 — `services/reranker.py`

```python
@lru_cache()
def get_reranker(): ...                      # 첫 질의에 지연 로드
def score(query: str, docs: List[Document]) -> List[float]
```

`sentence-transformers`의 `CrossEncoder`를 쓴다. **이미 설치된 의존성이다**
(3.3.1, 임베딩 모델이 쓰고 있다). 새 패키지가 필요 없다.

**점수는 0~1로 나온다.** `CrossEncoder.predict`는 `config.num_labels == 1`이면
`nn.Sigmoid()`를 자동 적용한다(`CrossEncoder.py:141`). `dragonkue/bge-reranker-v2-m3-ko`는
`id2label`이 한 개라 여기 해당하므로, 별도 정규화 없이 화면의 퍼센트 표시에
그대로 쓸 수 있다. 모델의 `max_position_embeddings`는 8194로 `chunk_size=1000`에
충분하다.

**torch 충돌은 새로 생기지 않는다.** API 프로세스에는 임베딩 모델 때문에 이미
torch가 올라와 있고, PaddleOCR는 그래서 이미 별도 프로세스로 격리돼 있다
(`ocr_isolate_process`). 리랭커는 이 구조를 바꾸지 않는다.

### 3.4 실패 처리

모델 로드 실패, GPU 메모리 부족, 추론 예외 — 어느 것이든 **로그를 남기고 벡터
순서 상위 `retrieval_k`개로 폴백한다. 질문은 실패시키지 않는다.**

리랭커가 없어도 답은 나온다. 답이 조금 나빠질 뿐이다. 이를 500 응답으로 바꾸는
것은 손해다.

### 3.5 비동기

`_get_relevant_documents`는 동기로 두고, `ainvoke`가 `BaseRetriever`의 스레드풀
폴백을 타는 현재 구조를 유지한다. 리랭킹도 그 안에서 돌아 이벤트 루프를 막지
않는다.

[LCEL 재작성 §3.4](2026-09-02-lcel-rewrite-design.md)와 같은 판단이다. 그 아래
chromadb가 동기이므로 `_aget_relevant_documents`를 추가해도 스레드풀로 내려가는
것은 같고, 검색 경로만 두 벌이 된다.

## 4. 범위

**포함**

- `services/reranker.py` 신설 (모델 소유, 점수 계산)
- `ScoringRetriever`에 리랭킹 단계 추가
- `_format_sources`의 정렬 기준을 `rerank_score` 우선으로
- `SourceDocument.rerank_score` 추가 및 프론트 표시
- 설정 4개 추가 및 `.env.example`·`README.md`·`ARCHITECTURE.md` 문서화
- 단위 테스트 (§7.1)

**제외**

- **리랭커 여러 개의 점수 합성.** RRF 등으로 복수 모델을 앙상블하는 방식.
  GPU 메모리와 지연이 배수로 늘고 합성 가중치를 맞춰야 한다. 단일 리랭커의
  효과를 먼저 관찰한 뒤 판단할 일이다.
- **단계적 필터링(캐스케이드).** 가벼운 리랭커로 좁힌 뒤 무거운 리랭커로 다시
  고르는 방식. 위와 같은 이유로 미룬다.
- **정량 평가셋.** 질문-정답 청크 쌍을 만들어 Recall@5로 채점하는 것. 재배치
  설계가 이미 별도 프로젝트로 분리했고, 이 작업에서도 그대로 둔다(§6).
- **`rerank_candidate_k` 값 튜닝.** 30이 적절한지는 A/B 관찰 뒤 판단한다.
- **재인덱싱.** 검색 단계만 바뀌므로 기존 문서를 다시 올릴 필요가 없다.
- **하이브리드 검색(BM25 앙상블).** 후보를 만드는 방식을 바꾸는 별개의 작업이다.

## 5. 설정

`config.py`에 추가한다.


| 키                   | 기본값                            | 의미                             |
| ---------------------- | ----------------------------------- | ---------------------------------- |
| `rerank_enabled`     | `True`                            | 끄면 지금과 동일하게 동작        |
| `rerank_model`       | `dragonkue/bge-reranker-v2-m3-ko` | 한국어 추가 학습된 다국어 리랭커 |
| `rerank_device`      | `cuda`                            | `cpu`, `cuda`, `cuda:0` …       |
| `rerank_candidate_k` | `30`                              | 리랭커에 넘길 후보 수            |

`retrieval_k`는 **의미를 바꾸지 않는다.** 지금도 앞으로도 LLM에 넣는 최종
개수이며, `.env`의 `RETRIEVAL_K=5`가 다른 뜻이 되는 일은 없다. 리랭커를 끄면
후보 = 최종이 되어 현재와 완전히 같아진다.

기본값을 `True`로 둔 것은 `retrieval_reorder`가 `True`로 출시된 선례를 따른
것이며, 로드되지 않는 2.27GB 경로를 남겨두지 않기 위해서다. **다만 §7.3의 A/B는
머지 전에 수행하고, 관찰 결과가 나쁘면 기본값을 뒤집는다.**

## 6. 한계

- **효과는 측정이 아니라 관찰이다.** 정답 셋이 없으므로 A/B는 육안 판단이다.
  개선이 관찰되지 않으면 그 사실도 그대로 기록한다.
- **질의당 지연이 늘어난다.** 후보 30개를 크로스인코더로 재채점하는 비용이 매
  질문에 붙는다. §7.3에서 함께 잰다.
- **GPU 메모리 2.27GB를 상주로 쓴다.** 현재 여유는 있으나 공짜가 아니다.
- **`rerank_candidate_k`가 클수록 좋다는 보장이 없다.** 후보를 넓힐수록 리랭커가
  잘못 건져 올릴 여지도 커진다. 30은 출발점이지 검증된 값이 아니다.
- **첫 질의가 느려진다.** 2.27GB 모델을 지연 로드하므로 첫 응답에 로딩 시간이
  포함된다. 임베딩 모델과 같은 성질이라 같은 방식으로 둔다.
- **`_format_sources`가 받는 순서가 재배치된 상태라는 가정은 여전하다.** 이
  작업이 그 결합을 없애지는 않는다.

## 7. 검증

### 7.1 단위 테스트

리랭커는 고정 점수를 돌려주는 대역으로 세운다. 실제 모델은 A/B에서만 쓴다.


| 테스트              | 지키는 것                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| 리랭크 점수 순 선택 | 후보 30개에서 상위 5개가 **리랭크 점수** 순으로 뽑히는지. 벡터 순서와 다른 정답을 세워 두 기준을 구분한다 |
| 끄면 동일           | `rerank_enabled=False`에서 후보 수가 `retrieval_k`가 되고 리랭커를 부르지 않는지                         |
| 예외 폴백           | 리랭커가 예외를 던져도 답변이 나오고, 벡터 순서 상위 N개가 오는지                                        |
| 두 점수 보존        | 재배치 후에도 `similarity_score`와 `rerank_score`가 모두 남는지                                           |
| 출처 정렬           | `_format_sources`가 리랭크 점수 순인지, 없을 때 벡터 점수로 되돌아가는지                                 |
| 후보 수 전달        | `rerank_candidate_k`가 실제로 Chroma의 `k`로 가는지 (켬/끔 양방향)                                       |

### 7.2 기존 테스트가 계약이다

`rerank_enabled=False`에서 기존 테스트가 **수정 없이** 통과해야 한다. 하나라도
손대야 한다면 끈 상태의 동작이 바뀐 것이므로 멈추고 이유를 확인한다.

`test_long_context_reorder.py`의 라이브러리 계약 테스트도 그대로 통과해야 한다.

### 7.3 수동 A/B (머지 전 필수)

`RERANK_ENABLED`를 `true`/`false`로 바꿔가며 같은 문서에 같은 질문을 던지고
비교한다.

- **출처 목록이 어떻게 달라지는지** — 리랭킹으로 새로 올라온 청크가 실제로 더
  관련 있는지
- **답변이 달라지는지** — 온도가 0이므로 컨텍스트가 같으면 답도 같다
- **질의당 지연** — 리랭킹이 더한 시간
- **첫 질의 로딩 시간** — 모델 지연 로드 비용

기대 효과는 "답이 정확해진다"가 아니라 **"벡터 검색이 5위 밖으로 밀어낸 청크가
구제되는 사례가 생긴다"** 이다. 관찰 결과를 트러블슈팅 문서 형식으로 남긴다.

개선이 관찰되지 않으면 `rerank_enabled` 기본값을 `False`로 뒤집고, 그 판단
근거를 기록한다.

## 8. 참고

- [검색 결과 재배치 설계 §4](2026-09-02-search-result-reordering-design.md) — 이 작업을 비대상으로 미뤄 둔 원문
- `sentence_transformers/cross_encoder/CrossEncoder.py:141` — `num_labels == 1`일 때 시그모이드 자동 적용
- `dragonkue/bge-reranker-v2-m3-ko` — 568M 파라미터, 가중치 2.27GB, `XLMRobertaForSequenceClassification`
