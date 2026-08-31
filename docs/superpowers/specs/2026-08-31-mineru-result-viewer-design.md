# MinerU 파싱 결과 뷰어

- 작성일: 2026-08-31
- 상태: 설계 승인됨, 구현 계획 대기

## 1. 문제

[MinerU 기반 PDF 파싱](2026-08-31-mineru-pdf-parsing-design.md) 도입 이후,
MinerU가 반환하는 블록 단위 결과(`content_list`: text/title/table/equation/image)는
`build_pages()`가 페이지 텍스트로 조립하는 즉시 소비되어 사라진다
(`mineru_client.py:216-275`). 조립된 페이지 텍스트는 청킹되어 Chroma에
임베딩으로만 남고, 원본 블록 구조(표의 HTML, 수식의 LaTeX, 어떤 이미지가
어느 페이지의 몇 번째 블록이었는지)는 어디에도 저장되지 않는다.

파싱 품질을 검증하거나 디버깅하려면 MinerU가 실제로 무엇을 인식했는지 봐야
하는데, 현재는 그럴 방법이 없다. 이 문서는 MinerU의 원본 파싱 결과를 문서별로
저장하고, 이를 조회하는 API와 프론트엔드 화면을 추가하는 설계를 다룬다.

## 2. 설계 원칙

- **읽기 전용, 파이프라인 비침습**: 이 기능은 검색/답변 파이프라인(청킹,
  임베딩, Chroma 저장)에 어떤 영향도 주지 않는다. 기존 `PdfPage` 조립 로직
  (`build_pages`)과 그 이후 단계는 무변경이다 — 원본 블록을 옆에서 한 번 더
  저장할 뿐이다.
- **최선형 부가 기능**: 저장이 실패해도 업로드/파싱은 절대 실패하지 않는다
  (기존 이미지 저장 실패 처리와 동일한 철학, `mineru_client.py:115-126`).
- **DB 신설 없음**: 이 저장소에는 관계형 DB가 없고(벡터스토어 Chroma뿐),
  문서 목록도 아직 없다(`documents.py:65-75`는 플레이스홀더). 새 기능을 위해
  DB를 들이는 대신, 기존 이미지 저장 방식과 동일하게 파일시스템에 JSON으로
  저장한다.
- **원본 그대로 노출**: "파싱 품질을 확인한다"는 목적에 맞춰 정제나 재해석
  없이 MinerU가 내려준 블록을 그대로 보여준다. LaTeX 렌더링, 마크다운 변환
  같은 가공은 하지 않는다(4절 비대상).

## 3. 설계

### 3.1 아키텍처 개요

```
DocumentProcessor.load_pdf()
        │
        ▼
MineruClient.parse_pdf() → MineruResult(blocks, images)
        │                         │
        │ (기존, 무변경)           │ (신규)
        ▼                         ▼
   build_pages()            parsed_store.save(document_id, filename, result)
        │                         │
        ▼                         ▼
   PdfPage 리스트           app/storage/parsed/{document_id}.json
   → 청킹/임베딩             app/storage/images/{document_id}/*.png (전체 이미지 블록)
   (기존 파이프라인)               │
                                  ▼
                    GET /api/v1/documents/parsed        (목록)
                    GET /api/v1/documents/{id}/parsed    (상세)
                                  │
                                  ▼
                    프론트엔드 "문서 뷰어" 탭
```

### 3.2 백엔드 — 원본 결과 저장

`mineru_client.parse_and_build_pages()`에 선택적 콜백 `on_parsed`를 추가한다:

```python
def parse_and_build_pages(
    file_path: str,
    ocr: Optional[SupportsImageOcr],
    image_dir: Optional[Path] = None,
    client: Optional[MineruClient] = None,
    on_parsed: Optional[Callable[[MineruResult], None]] = None,
) -> list:
    if client is None:
        client = MineruClient()
    result = client.parse_pdf(file_path)
    if on_parsed is not None:
        on_parsed(result)
    return build_pages(result.blocks, result.images, ocr, image_dir)
```

`on_parsed`는 `build_pages()`가 블록을 소비하기 **전**, 원본 `MineruResult`
(블록 + base64 이미지 딕셔너리)에 접근할 수 있는 유일한 지점이다. 예외를
전파하지 않는 것은 호출자(`parsed_store.save`) 책임으로 둔다 — 클라이언트
모듈 자체는 저장 방식을 모른다(관심사 분리).

새 모듈 **`backend/app/services/parsed_store.py`**가 저장을 담당한다:

```python
def save(document_id: str, filename: str, result: MineruResult) -> None:
    """MinerU 원본 블록을 문서별 JSON으로 저장한다. 절대 예외를 전파하지
    않는다 — 실패는 로그만 남기고 업로드 흐름을 막지 않는다."""
```

동작:
- `result.blocks`를 `page_idx`로 그룹핑한다(순서는 이미 리딩 오더).
- `img_path`를 가진 모든 블록(현재 `build_pages`는 OCR 대상 이미지만
  저장하지만, 뷰어는 표/수식의 스크린샷을 제외한 순수 `image` 타입 블록도
  모두 보여줘야 하므로 **전체** `image` 타입 블록의 이미지를 저장 대상으로
  삼는다)의 이미지를 `image_storage_dir/{document_id}/{md5}.png`로 저장한다.
  기존 `_save_image`/이미지 ID 스킴(bytes의 md5 hexdigest[:16])을 재사용해
  기존 이미지 서빙 엔드포인트와 충돌 없이 공유한다.
- 블록을 직렬화 가능한 형태로 변환한다: `type`, 타입별 본문(`text`/
  `table_body`/LaTeX `text`), `image` 타입이면 `image_id`.
- 최종 구조를 `parsed_storage_dir/{document_id}.json`에 기록한다:

```json
{
  "document_id": "doc_...",
  "filename": "example.pdf",
  "created_at": "2026-08-31T12:00:00+00:00",
  "page_count": 12,
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {"type": "title", "text": "..."},
        {"type": "text", "text": "..."},
        {"type": "table", "table_body": "<table>...</table>"},
        {"type": "equation", "text": "E=mc^2"},
        {"type": "image", "image_id": "1158b6dbecf37fc3"}
      ]
    }
  ]
}
```

`document_processor.load_pdf()`는 `parse_and_build_pages()` 호출에
`on_parsed=lambda result: parsed_store.save(document_id, filename, result)`를
넘긴다(단, `document_id`가 있을 때만 — 이미지 저장과 동일한 전제).
`parsed_store.save` 내부에서 모든 예외를 잡아 로그만 남긴다.

MinerU가 비활성이거나(`mineru_enabled=False`) 파싱이 실패해
`PyPDFLoader` 폴백 경로를 타는 문서는 `on_parsed`가 호출되지 않으므로
파싱 JSON 자체가 없다 — 조회 API는 이 경우 404를 반환한다(3.3절).

### 3.3 백엔드 — 조회 API

`backend/app/api/routes/documents.py`에 두 엔드포인트를 추가한다:

- `GET /api/v1/documents/parsed` → `List[ParsedDocumentSummary]`
  - `parsed_storage_dir`의 모든 `*.json` 파일을 스캔해 `document_id`,
    `filename`, `created_at`, `page_count`만 읽어 목록으로 반환한다
    (매번 전체 페이지/블록까지 파싱할 필요 없음 — 상위 필드만 읽는다).
- `GET /api/v1/documents/{document_id}/parsed` → `ParsedDocumentDetail`
  - 해당 문서의 JSON 파일 전체를 읽어 반환. 파일이 없으면 404
    (`"Parsed result not found"`).
  - `document_id`는 기존 `_SAFE_ID` 화이트리스트로 검증해 경로 조작을
    막는다(`resolve_image_path`와 동일 패턴).

응답 모델(`app/api/models/responses.py`에 추가):

```python
class ParsedBlock(BaseModel):
    type: str
    text: Optional[str] = None
    table_body: Optional[str] = None
    image_id: Optional[str] = None

class ParsedPage(BaseModel):
    page_number: int
    blocks: List[ParsedBlock]

class ParsedDocumentSummary(BaseModel):
    document_id: str
    filename: str
    created_at: str
    page_count: int

class ParsedDocumentDetail(ParsedDocumentSummary):
    pages: List[ParsedPage]
```

기존 `GET /api/v1/documents/`(문서 목록 플레이스홀더)와 `DELETE
/api/v1/documents/{document_id}`는 이 설계에서 건드리지 않는다 — 별개
관심사(전체 문서 관리 vs. 파싱 결과 열람)이며, 문서 삭제 시 파싱 JSON을
함께 지우는 것은 범위 밖으로 둔다(5절 한계).

### 3.4 프론트엔드 — 뷰어 화면

`react-router` 등 라우팅 라이브러리가 없는 단일 화면 앱이므로, 새 페이지
대신 `App.tsx`에 로컬 state 기반 탭 토글을 추가한다: `activeTab: 'chat' |
'viewer'`. 헤더에 탭 버튼 두 개, `viewer` 선택 시 기존
`FileUploader`+`ChatWindow` 대신 새 컴포넌트를 렌더링한다.

새 컴포넌트 `frontend/src/components/ParsedDocumentViewer.tsx` +
`ParsedDocumentViewer.css`:

- 마운트 시 `apiService.getParsedDocuments()`로 목록을 불러와 좌측 사이드바에
  표시(파일명, 페이지 수). 파싱 JSON이 없는 문서는 애초 목록에 나타나지 않음
  (백엔드가 존재하는 파일만 나열하므로 자연히 필터링됨).
- 문서를 클릭하면 `apiService.getParsedDocument(documentId)`로 상세를 불러와
  우측 패널에 페이지별로 렌더링:
  - `type: "title"/"text"` → 문단(`title`은 약간 굵게)
  - `type: "table"` → `table_body`(HTML)를 **DOMPurify로 sanitize한 후**
    `dangerouslySetInnerHTML`로 렌더링. MinerU가 업로드된 PDF의 텍스트로부터
    생성한 HTML을 그대로 주입하는 것은 XSS 벡터이므로 신규 의존성
    `dompurify`(+ `@types/dompurify`)를 추가해 sanitize 후에만 렌더링한다.
  - `type: "equation"` → LaTeX 원문을 `<pre><code>`로 표시(렌더링 없음 —
    4절 비대상).
  - `type: "image"` → 기존 이미지 서빙 엔드포인트를 그대로 사용:
    `<img src={`/api/v1/documents/${documentId}/images/${block.image_id}`} />`.
- API 타입은 `frontend/src/types/api.types.ts`에 `ParsedBlock`,
  `ParsedPage`, `ParsedDocumentSummary`, `ParsedDocumentDetail`을 백엔드
  응답 모델과 1:1로 추가한다.
- `frontend/src/services/api.ts`의 `ApiService`에 `getParsedDocuments()`,
  `getParsedDocument(documentId)` 메서드를 기존 `getDocuments()`와 같은
  패턴(axios, `DEFAULT_TIMEOUT`)으로 추가한다.

### 3.5 설정 추가

| 키 | 기본값 | 비고 |
|---|---|---|
| `parsed_storage_dir` | `"app/storage/parsed"` | 파싱 JSON 저장 경로. `image_storage_dir`와 같은 패턴 |

## 4. 범위

### 대상
- MinerU 원본 블록(텍스트/제목/표/수식/이미지)을 문서별로 영속화한다.
- 파싱된 문서 목록과 페이지별 블록 상세를 조회하는 API 2개를 추가한다.
- 프론트엔드에 파싱 결과를 열람할 수 있는 뷰어 화면(탭)을 추가한다.

### 비대상 (YAGNI)
- LaTeX/수식 렌더링(KaTeX 등) — 원본 텍스트 노출로 충분, 필요해지면 별도 요청.
- 문서 삭제(`DELETE /documents/{id}`) 시 파싱 JSON 동반 삭제 — 기존 삭제
  엔드포인트도 이미지 디렉터리 삭제 외엔 정리 로직이 제한적이라, 이 설계도
  같은 수준에 맞춘다. 파싱 JSON 잔존은 이후 별도 정리 작업으로 다룬다.
- 페이지네이션/검색/필터 — 목록 API는 전체를 한 번에 반환한다. 문서 수가
  많아지면 필요해질 수 있으나 현재 이 앱의 사용 규모에서는 불필요.
- 기존 `GET /api/v1/documents/` 플레이스홀더 구현 — 별개 관심사, 이 설계로
  해결하지 않는다.
- 회귀 방지용 골든 파일 비교, 파싱 결과 diff 등 고급 QA 도구 — 지금은
  "눈으로 확인" 수준의 뷰어만 필요하다.

## 5. 한계

- 파싱 JSON은 문서 삭제와 별도로 관리되므로, `DELETE /documents/{id}` 이후에도
  `parsed_storage_dir`에 파일이 남는다(4절 비대상). 디스크 정리가 필요해지면
  별도 작업으로 다룬다.
- 목록 API가 매 요청마다 모든 JSON 파일을 열어 상위 필드만 읽으므로, 문서
  수가 매우 많아지면(수천 건) 응답이 느려질 수 있다 — 현재 규모에서는
  허용 가능한 트레이드오프로 판단한다.
- 표/수식은 원본 HTML/LaTeX 그대로 노출되므로, 프론트엔드에서 시각적으로
  "예쁘게" 보이지는 않는다(표는 브라우저 기본 스타일, 수식은 텍스트) — 이는
  의도된 것이다(3절 설계 원칙: 원본 그대로 노출).
- `_serialize_block`은 `type`/`text`/`table_body`/`image_id`만 옮기고,
  MinerU `content_list` 블록이 실제로 가질 수 있는 `text_level`(제목 단계),
  `image_caption`/`image_footnote`, `table_caption`/`table_footnote`(각각
  [MinerU 파싱 설계 문서](2026-08-31-mineru-pdf-parsing-design.md)의 3.2절
  표에 문서화됨)는 옮기지 않는다. "원본 그대로 노출"이라는 3절 설계 원칙에
  비추면 이 필드들도 보여주는 편이 더 충실하겠지만(예: 잘못 파싱된 도표의
  캡션을 확인할 수 없음), 4개 계층(직렬화/응답 모델/TS 타입/렌더링)에 걸친
  실제 작업량을 고려해 최초 구현 범위에서는 의도적으로 제외했다(final
  whole-branch review에서 지적, 2026-08-31). 필요해지면 별도 작업으로
  다룬다.
