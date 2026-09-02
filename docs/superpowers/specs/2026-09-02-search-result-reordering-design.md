# 검색 결과 재배치 (Lost in the Middle 완화)

- 작성일: 2026-09-02
- 상태: 설계 승인됨, 구현 계획 대기
- 관련 코드: `rag_service.py`, `config.py`
- 선행 작업: [검색 결과가 머리글 조각으로만 채워지는 문제](../../troubleshooting/2026-09-01-rag-retrieval-quality.md)

## 1. 문제

[RAG 검색 품질 트러블슈팅](../../troubleshooting/2026-09-01-rag-retrieval-quality.md)에서
인덱싱 단계의 세 가지 결함(페이지 부속물 오염, 페이지 경계에 갇힌 청크, Document별
백분위 임계값)을 고쳤다. 그 문서의 결론은 "증상은 검색 단계에 나타났지만 원인은
전부 인덱싱 단계에 있었다"였고, **검색 단계 자체는 손대지 않았다.**

이 문서는 그 다음 단계를 다룬다.

`ScoringRetriever`는 Chroma가 돌려준 청크 `retrieval_k`개를 관련성 내림차순 그대로
`StuffDocumentsChain`에 넘긴다. 즉 가장 관련성 높은 청크가 컨텍스트 맨 앞에, 가장
낮은 청크가 맨 뒤에 놓인다.

문제는 LLM이 긴 컨텍스트를 균일하게 읽지 않는다는 것이다. Liu et al. 2023
["Lost in the Middle"](https://arxiv.org/abs/2307.03172)은 정답이 컨텍스트 중앙에
놓였을 때 정확도가 눈에 띄게 떨어지고, 시작과 끝에서 높아지는 U자 곡선을 보고했다.

현재 배치에서는 **관련성 2~9위 청크가 전부 이 취약 구간에 들어간다.**
`chunk_size=1000`, `retrieval_k=10`이므로 컨텍스트는 약 1만 자에 달하고, 이 곡선이
의미를 갖기에 충분히 길다.

## 2. 설계 원칙

- **검색 결과의 내용과 개수는 바꾸지 않는다.** 이 작업은 순수하게 청크의 **순서**만
  다룬다. 어떤 청크를 고를지는 벡터 검색이 그대로 결정한다.
- **재인덱싱 불필요.** 변경은 검색 단계에만 있으므로 기존 문서를 다시 올릴 필요가
  없다. 인덱싱 단계를 고쳤던 청킹 수정과 대비되는 지점이다.
- **알고리즘을 소유하지 않는다.** `langchain-community`가 제공하는
  `LongContextReorder`를 그대로 쓴다. 이미 설치돼 있고, 직접 구현하면 유지·테스트
  대상이 늘어난다.
- **내부 최적화가 사용자 UI로 새어나가지 않는다.** 재배치는 LLM을 위한 것이므로,
  사용자에게 보이는 출처 목록은 관련성 순서를 유지한다.
- **끌 수 있어야 한다.** 효과를 사례로 관찰하려면 같은 질문을 켠 상태와 끈 상태로
  던질 수 있어야 한다.

## 3. 설계

### 3.1 데이터 흐름

```
similarity_search_with_relevance_scores(query, k, filter)
        │  [(doc, score), ...]  ← Chroma가 관련성 내림차순 보장
        ▼
doc.metadata["similarity_score"] = score        ← 반드시 재배치 "전"
        │
        ▼
LongContextReorder().transform_documents(docs)  ← retrieval_reorder=True일 때만
        │
        ├──► 체인 컨텍스트 (재배치된 순서 = LLM이 보는 순서)
        │
        └──► return_source_documents
                    │
                    ▼
             _format_sources()
             similarity_score 내림차순 재정렬
                    │
                    ▼
             사용자에게 보이는 출처 목록
```

### 3.2 재배치 지점 — `ScoringRetriever`

`_get_relevant_documents` 안에서 처리한다. 검색·점수 부착·재배치가 한 메서드에
모여 있어 순서 의존성이 눈에 보인다.

`LongContextReorder`는 상태 없는 pydantic 모델이므로 `QA_PROMPT`와 마찬가지로
**모듈 레벨에서 한 번** 생성한다. 요청마다 만들 이유가 없다.

`transform_documents()`의 반환 타입은 `Sequence[Document]`이고
`_get_relevant_documents`는 `List[Document]`를 반환해야 하므로 `list(...)`로 감싼다.

리트리버에 `reorder: bool` 필드를 두고, `RAGService.ask_question`이 생성 시점에
`settings.retrieval_reorder`를 주입한다. 리트리버 자체는 설정 모듈을 직접 읽지
않으므로 테스트에서 두 조건을 모두 세울 수 있다.

### 3.3 점수는 재배치 전에 심는다

이 순서가 설계의 핵심이다.

`similarity_search_with_relevance_scores`는 `(doc, score)` 쌍의 리스트를 준다.
재배치하면 문서 순서가 바뀌면서 **`docs`와 `scores` 리스트의 짝이 깨진다.** 재배치
후에는 어떤 점수가 어떤 문서의 것인지 복원할 방법이 없다.

점수를 먼저 `doc.metadata["similarity_score"]`에 실어 두면 값이 문서 객체를
따라다니므로, 재배치 뒤에도 `_format_sources`가 점수순으로 되돌릴 수 있다. 이는
기존 `ScoringRetriever`가 이미 하던 일이며, 재배치는 그 **뒤에** 붙는다.

### 3.4 출처 목록은 점수순으로 되돌린다 — `_format_sources`

`ConversationalRetrievalChain`은 리트리버가 돌려준 리스트를 컨텍스트 조립과
`source_documents` 양쪽에 그대로 쓴다. 따라서 재배치된 순서가 그대로 API 응답에
실리고, 프론트엔드([`Message.tsx`](../../../frontend/src/components/Message.tsx))는
배열 순서대로 렌더링하므로 그대로 화면에 나간다.

이 경우 사용자는 `Relevance 62%`가 `Relevance 91%`보다 위에 뜨는 화면을 보게 되고
고장으로 오해한다. 재배치는 LLM을 위한 내부 최적화이므로 UI까지 새어나갈 이유가
없다.

`_format_sources`가 `similarity_score` 내림차순으로 정렬한 뒤 포매팅한다.
`similarity_score`가 `None`인 문서(리트리버를 거치지 않고 들어온 경우)는 `0.0`으로
취급해 정렬이 깨지지 않게 한다.

### 3.5 프롬프트 문구 수정 — `QA_PROMPT`

현재 프롬프트에는 다음 헤더가 있다.

```
## 검색된 문서 (관련성 순)
```

재배치 후에는 사실이 아니다. 그대로 두면 LLM에게 "앞쪽 문서가 더 관련성이 높다"는
**틀린 힌트**를 주게 되며, 이는 재배치가 노리는 효과와 정면으로 충돌한다.
`(관련성 순)`을 제거한다.

### 3.6 홀짝에 따른 방향 차이

`LongContextReorder`의 내부 구현(`_litm_reordering`)은 `documents.reverse()`로
시작한다. 그 결과 입력 개수의 홀짝에 따라 배치 방향이 달라진다.

| 입력 개수 | 출력 (1이 가장 관련성 높음) | 1등 위치 |
|---|---|---|
| 5 | `1 3 5 4 2` | 맨 앞 |
| 7 | `1 3 5 7 6 4 2` | 맨 앞 |
| 4 | `2 4 3 1` | 맨 뒤 |
| **10** | **`2 4 6 8 10 9 7 5 3 1`** | **맨 뒤** |

**어느 쪽이든 "관련성 높은 청크는 양 끝, 낮은 청크는 가운데"라는 목표 자체는 정확히
만족한다.** 짝수일 때는 홀수일 때 출력의 역순일 뿐이며, 버그가 아니다.

`retrieval_k`가 10(짝수)이므로 이 프로젝트에서는 **1등 청크가 컨텍스트 맨 뒤,
2등이 맨 앞**에 놓인다. Liu et al.의 U자 곡선에서 시작 위치가 끝 위치보다 대체로
유리하므로 약간 불리한 선택이지만, 알고리즘을 직접 소유하지 않는 편익과 맞바꾼
것으로 기록한다. 6절에 한계로 명시한다.

### 3.7 오류 처리

없다. `transform_documents`는 순수한 리스트 재배치이고 I/O도 외부 호출도 없다.
빈 리스트와 단일 원소 리스트는 그대로 통과한다. 검색이 0건이면 재배치도 0건이다.

## 4. 범위

### 대상

- `ScoringRetriever`에 `reorder` 필드 추가 및 `LongContextReorder` 적용
- `_format_sources`의 점수순 정렬
- `QA_PROMPT`의 `(관련성 순)` 제거
- `retrieval_reorder` 설정 추가 및 `.env.example` 문서화
- 단위 테스트 및 라이브러리 계약 테스트
- `README.md`의 설정 표(191행, 영문)에 새 항목 반영

### 비대상 (YAGNI)

- **크로스인코더 리랭킹.** k를 20~30으로 늘려 뽑고 리랭커 모델로 재채점해 상위 N개만
  남기는 방식. 검색 품질 개선폭은 크지만 모델 로딩·GPU 메모리·질의당 지연이
  추가되며, 이는 "무엇을 고를지"를 바꾸는 별개의 작업이다.
- **`retrieval_k` 값 튜닝.** 10이 적절한지는 이 작업에서 판단하지 않는다.
- **정량 평가셋 구축.** 질문-정답 청크 쌍을 만들어 자동 채점하는 것은 이 기능
  구현보다 훨씬 큰 작업이므로 별도 프로젝트로 분리한다.
- **비동기 리트리버 경로.** 현재 체인은 동기 호출이다.

## 5. 설정

`config.py`:

```python
retrieval_reorder: bool = True
```

`.env.example`:

```
RETRIEVAL_REORDER=true  # 관련성 높은 청크를 컨텍스트 앞/뒤 끝에 배치 (Lost in the Middle 완화)
```

기본값은 켠 상태다. 끄면 기존 동작(관련성 내림차순 그대로)으로 돌아간다.

## 6. 한계

- **`retrieval_k=10`은 짝수라 1등 청크가 컨텍스트 맨 뒤에 놓인다.** `RETRIEVAL_K`를
  홀수로 바꾸면 배치 방향이 조용히 뒤집힌다. 이 결합은 `LongContextReorder`를 그대로
  쓰기로 한 결정에서 나온 것이며, 7.2절의 계약 테스트가 이 동작을 고정한다.
- **`atransform_documents`는 `NotImplementedError`를 던진다.** 현재 체인은 동기
  경로라 무관하지만, 비동기 리트리버로 전환하면 이 지점에서 실패한다.
- **개선 효과는 측정된 것이 아니라 사례로 관찰한 것이다.** 정답 셋이 없으므로
  A/B 비교는 육안 판단이다. 청킹 수정 때처럼 청크 길이 분포 같은 객관적 지표로
  검증할 수 있는 성질의 변경이 아니다.
- **`langchain-community==0.0.38` 핀에 묶여 있다.** 상위 버전에서 이 클래스의 import
  경로가 이동했으므로, 의존성을 올릴 때 함께 손봐야 한다.
- **점수 순서를 Chroma에 의존한다.** `LongContextReorder`는 점수를 보지 않고 입력이
  이미 관련성 순으로 정렬돼 있다고 가정한다. 현재는
  `similarity_search_with_relevance_scores`가 보장하지만, 나중에 리트리버와 재배치
  사이에 필터링이나 병합 단계가 끼면 이 가정이 조용히 깨진다.

## 7. 검증

### 7.1 단위 테스트

| 테스트 | 지키는 것 |
|---|---|
| 리트리버 재배치 순서 | `k=10`에서 출력이 `2 4 6 8 10 9 7 5 3 1`인지 고정 |
| 점수 보존 | 재배치 후에도 각 문서의 `similarity_score`가 유지되는지 |
| `reorder=False` | 플래그를 끄면 관련성 내림차순 그대로인지 |
| 빈 결과 | 검색 0건에서 터지지 않는지 |
| `_format_sources` 정렬 | 재배치된 입력을 받아도 출처가 점수 내림차순인지, `None` 점수를 방어하는지 |

### 7.2 라이브러리 계약 테스트

`LongContextReorder`의 홀수/짝수 방향 동작을 명시적으로 고정하는 테스트를 둔다.

우리는 서드파티의 **정확한 순서 동작**에 의존하기로 했다. 이 테스트가 없으면
`langchain-community` 버전을 올렸을 때 배치가 바뀌어도 아무도 모른 채 넘어간다.
3.6절 표의 값을 그대로 단언한다.

### 7.3 수동 A/B

`RETRIEVAL_REORDER`를 `true`/`false`로 바꿔가며 같은 문서에 같은 질문 몇 개를 던져
답변을 육안으로 비교하고, 관찰 결과를 트러블슈팅 문서 형식으로 남긴다.

기대 효과는 "재배치가 답을 정확하게 만든다"가 아니라 **"핵심 청크가 중앙에 묻혀
누락되던 사례가 줄어든다"** 이다. 개선이 관찰되지 않으면 그 사실도 그대로 기록한다.
