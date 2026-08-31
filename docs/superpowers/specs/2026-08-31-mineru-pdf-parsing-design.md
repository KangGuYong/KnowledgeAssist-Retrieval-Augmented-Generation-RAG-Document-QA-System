# MinerU 기반 PDF 파싱 도입

- 작성일: 2026-08-31
- 상태: 설계 승인됨, 구현 계획 대기

## 1. 문제

현재 PDF 처리는 `pdf_ocr.py`가 PyMuPDF(fitz)로 페이지 레이아웃을 직접 분석한다:
텍스트/이미지 블록을 분류하고, 좌표 기반 휴리스틱(`ocr_row_tolerance`,
`ocr_layout_order`)으로 읽기 순서를 근사하고, 이미지가 너무 많거나
(`ocr_max_images_per_page`) 텍스트가 거의 없으면(`ocr_page_text_threshold`)
스캔 문서로 간주해 전체 페이지를 한 번에 OCR한다. 이 휴리스틱들은 모두 자체
구현이라 유지보수 부담이 있고, 세 가지 구체적 품질 문제가 있다:

- **읽기 순서/레이아웃 정확도**: 다단 레이아웃, 헤더/푸터, 캡션이 섞인 복잡한
  문서에서 좌표 정렬만으로는 자연스러운 순서를 보장하지 못한다.
- **표**: 표는 이미지 블록으로 렌더링돼 PaddleOCR에 그대로 넘어간다. PaddleOCR은
  줄 단위 텍스트 인식기라 표의 셀 구조(행/열 경계)를 전혀 보존하지 못하고, 셀
  텍스트가 뒤섞인 한 덩어리 문자열로 나온다.
- **수식**: 별도 처리가 없다 — 수식이 포함된 이미지 블록은 PaddleOCR이 텍스트
  인식을 시도하지만 LaTeX/수학 표기를 인식하도록 만들어지지 않았다.

MinerU(오픈소스 문서 파싱 도구)는 레이아웃 분석, 표(HTML 변환), 수식(LaTeX 변환),
읽기 순서 결정을 모델 기반으로 통합 처리한다. 이 문서는 MinerU를 PDF 파싱의
1차 경로로 도입하는 설계를 다룬다.

## 2. 설계 원칙

- **완전 대체, 부분 재사용**: PyMuPDF 기반 레이아웃 분석(`pdf_ocr.py`)은 MinerU로
  완전히 대체한다. 단, PaddleOCR 엔진 자체(`ocr_service.py`, `ocr_worker.py`,
  `ocr_subprocess.py`)는 그대로 남는다 — MinerU가 도표/차트 이미지 **내부**의
  텍스트까지는 자동으로 읽어주지 않으므로, 그 역할은 계속 PaddleOCR이 맡는다
  (3.3절).
- **기존 하위 파이프라인 무변경**: `PdfPage` 스키마, `document_processor.py`의
  `chunk_documents()`/`process_file()`, `chunking.py`, `rag_service.py`,
  이미지 서빙 엔드포인트(`documents.py`)는 이 설계로 인해 수정되지 않는다.
  MinerU 도입은 `load_pdf()`가 페이지 텍스트를 만드는 **방법**만 바꾼다.
- **장애 시 텍스트 추출은 계속된다**: MinerU 서비스가 죽어 있어도 업로드는
  실패하지 않는다 (기존 `ocr_enabled=False`/OCR 실패 시 `PyPDFLoader` 폴백과
  동일한 철학).

## 3. 설계

### 3.1 아키텍처 개요

```
Upload → DocumentProcessor.load_pdf()
            │
            ▼
   MineruClient.parse_pdf(file_path)  ──HTTP──▶  MinerU 서비스
            │                                     (DGX 자체 호스팅, mineru-api)
            │  content_list.json(블록 목록) + images/
            ▼
   page_idx로 블록을 페이지 단위로 그룹핑
   type="image" 블록 → PaddleOCR(get_ocr_service)로 텍스트 보강
   type="table"/"equation" 블록 → table_body(HTML)/text(LaTeX) 그대로 삽입
            │
            ▼
   기존과 동일한 PdfPage 리스트로 변환 → 이후 파이프라인은 무변경
```

### 3.2 MinerU 서비스 배포

MinerU 패키지가 제공하는 FastAPI 서버(`mineru-api`)를 이 DGX 머신에 백엔드와는
별도의 상시 프로세스로 구동한다. 포트는 기존 서비스(백엔드 8000, 프런트 5173,
Ollama 11434, ComfyUI 8188, open-webui 8080)와 겹치지 않는 값을 쓴다.

GPU는 이미 ComfyUI가 점유하고 있는 자원을 공유하게 된다 — 서비스 기동 시점의
실제 GPU 메모리 여유는 구현 단계에서 실측해 확인한다(설계로 미리 예단하지
않는다).

`content_list.json`의 각 블록은 다음 필드를 갖는다(공통: `type`, `page_idx`,
`bbox`):

| type | 관련 필드 |
|---|---|
| `text` | `text`, `text_level`(있으면 헤딩) |
| `image` | `img_path`, `image_caption`, `image_footnote` |
| `table` | `img_path`(스크린샷), `table_body`(HTML), `table_caption`, `table_footnote` |
| `equation` | `img_path`, `text`(LaTeX), `text_format` |

**실측 확인됨** (2026-08-31, 설치된 MinerU 3.4.5, `mineru-api --port 8100`):
`/file_parse`에 실제 요청을 보내 응답 형태를 확인했다 — 이전 버전의 "가정"
문단(같은 호스트 파일시스템을 공유한다는 가정)은 **틀렸다.** 실제로는:

- 요청은 `multipart/form-data`, 파일 필드명은 `files`(복수, 배열 — 단일
  파일 업로드도 `files=`로 보낸다). `backend` 폼 필드의 **기본값은
  `hybrid-engine`**(로컬 VLM 필요)이라 **반드시 명시적으로
  `backend=pipeline`을 보내야** 2절의 "완전 대체, 부분 재사용" 원칙(VLM
  백엔드는 비대상)이 지켜진다. `return_content_list=true`,
  `return_images=true`도 기본값이 `false`라 명시적으로 켜야 한다.
  `lang_list`의 기본값은 `["ch"]`(중국어)라, 이 앱의 문서가 한국어이므로
  `lang_list=korean`을 보낸다("Korean, English" 인식).
- 응답은 `content_list.json`을 감싼 **작업(task) 봉투**다:
  `{"task_id", "status", "error", "file_names": ["sample"], "results":
  {"sample": {"content_list": "<JSON 문자열>", "images": {...}}}}`.
  `results`는 업로드 파일명의 확장자 없는 스템으로 키가 잡히므로,
  `file_names[0]`으로 조회한다. `/file_parse`는 동기 엔드포인트라
  `status`는 이미 `"completed"`(또는 실패 시 `"failed"`+`error`)로 와서
  별도 폴링이 필요 없다.
- `content_list`는 (블록 배열이 아니라) **그 배열을 담은 JSON 문자열**이라
  한 번 더 `json.loads()`해야 한다.
- `images`는 `{img_path: "data:image/jpeg;base64,..."}` 형태의 **base64
  데이터 URI 딕셔너리**다 — 파일시스템 경로가 전혀 아니다. 즉 "같은 호스트
  파일시스템 공유" 가정은 필요 없다: 이미지 바이트가 응답 JSON 안에 직접
  들어온다. MinerU 서비스가 백엔드와 다른 호스트에 있어도 이 설계는 그대로
  동작한다(장점이자, 원래 가정이 틀렸던 이유).

3.3~3.4절의 구현은 이 실측 결과를 반영한다: `MineruResult`는
`output_dir: Path` 대신 `images: dict[str, str]`(img_path → base64 데이터
URI)을 담고, 이미지 블록 처리는 파일을 여는 대신 이 딕셔너리에서
`img_path`를 찾아 base64 디코드한다.

### 3.3 백엔드 통합 모듈

`pdf_ocr.py`의 PyMuPDF 레이아웃 로직을 새 모듈
**`backend/app/services/mineru_client.py`**로 대체한다.

```python
def parse_pdf(file_path: str) -> MineruResult:
    """MinerU 서비스(/file_parse)를 호출해 content_list 블록과 이미지
    (img_path -> base64 데이터 URI)를 받아온다. backend=pipeline,
    lang_list=korean, return_content_list=true, return_images=true를
    명시적으로 보낸다(서버 기본값은 hybrid-engine/false/["ch"]). 응답의
    status가 completed가 아니거나 error가 있으면, 혹은 연결 오류/타임아웃/
    5xx면 예외를 그대로 전파한다 — 폴백 판단은 호출자(document_processor.
    load_pdf)의 책임."""
```

`document_processor.load_pdf()`는 `extract_pages()` 호출을 `mineru_client`
경로로 교체하되, 기존 try/except 폴백 구조(3.6절)는 그대로 유지한다.

**페이지 재구성**: `content_list.json`은 `page_idx`(0-based) 기준 평평한
블록 리스트다. 이를 `page_idx`로 그룹핑해 페이지별로 블록을 순서대로 이어붙인다
(이미 리딩 오더로 정렬되어 반환되므로 추가 정렬 불필요).

- `text` 블록: `text` 필드를 그대로 이어붙인다.
- `table` 블록: `table_body`(HTML)를 텍스트에 삽입한다. HTML 태그를 그대로
  넣을지 셀 텍스트만 뽑아 넣을지는 임베딩 검색 품질에 달려 있어 설계로 미리
  결정하지 않는다 — 구현 시 실측 후 선택한다(범위 밖 아님, 이 기능의 일부다).
- `equation` 블록: `text`(LaTeX)를 그대로 삽입한다.
- `image` 블록: 3.4절의 OCR 보강 결과로 치환한다.

**`PdfPage` 매핑**은 기존 스키마를 그대로 채운다:

```python
PdfPage(
    page_number=page_idx + 1,
    text=<페이지의 블록들을 순서대로 결합한 텍스트>,
    ocr_image_count=<해당 페이지에서 텍스트를 얻은 image 블록 수>,
    image_ids=[...],
)
```

`full_page_ocr` 필드는 PyMuPDF 경로 전용 개념(스캔 페이지 여부를 이 앱이 직접
판단)이었다 — MinerU 경로에서는 그 판단 자체를 MinerU가 내부적으로 하므로
이 앱에서는 알 수 없다. `PdfPage.full_page_ocr`는 MinerU 경로에서 항상
`False`로 채운다(필드 제거는 하지 않는다 — `document_processor.py`가
메타데이터로 그대로 내보내고 있어, 제거하면 그 소비처까지 손대야 한다. 이
설계의 범위 밖이므로 값만 고정한다).

### 3.4 이미지 처리 & 인용 기능 유지

`image` 블록마다 `images[block["img_path"]]`(base64 데이터 URI)를 디코드해
numpy 배열로 변환하고, 기존 `get_ocr_service().image_to_text(image)`를 그대로
호출한다. PaddleOCR 서브프로세스 격리(`ocr_isolate_process`,
`ocr_subprocess.py`)는 수정하지 않는다.

```python
for block in image_blocks_on_page:
    data_uri = images[block["img_path"]]  # "data:image/jpeg;base64,...."
    raw = load_image_as_bgr_array_from_data_uri(data_uri)
    text = ocr.image_to_text(raw)
    if text:
        page_text_parts.append(_format_ocr_block(text))  # 기존 [이미지 텍스트] 마커 재사용
```

**`image_id` 스킴은 기존과 동일하게 유지**한다: 이미지 바이트의
`md5(...).hexdigest()[:16]`을 `image_id`로 쓰고, `{image_storage_dir}/
{document_id}/{image_id}.png`로 PNG 통일 저장한다. 이렇게 하면
`documents.py`의 `resolve_image_path`(`_SAFE_ID` 화이트리스트),
`/documents/{id}/images/{image_id}` 서빙 엔드포인트, `_format_sources`의
`image_ids` 콤마 조인, 프런트엔드 썸네일/라이트박스 — 이 전부가 코드 변경
없이 그대로 동작한다.

`table`/`equation` 블록의 `img_path`(스크린샷)는 인용 이미지로 저장하지
**않는다** — 이미 구조화된 텍스트(HTML/LaTeX)로 본문에 들어가 있으므로
원본 스크린샷을 별도로 보여줄 필요가 3.1(구 설계)절이 다루던 "OCR 오인식
방지" 동기에 해당하지 않는다.

### 3.5 장애 대응

`load_pdf()`의 기존 try/except 폴백 구조를 그대로 쓴다 — 호출부만 바뀐다:

```python
try:
    pages = mineru_client.parse_and_build_pages(file_path, image_dir)
except Exception as e:
    logger.warning(f"MinerU extraction failed for {filename} ({e}); falling back to text-only extraction")
    return PyPDFLoader(file_path).load()
```

MinerU 서비스가 다운되어 있거나, 특정 PDF에서 파싱 오류를 내거나, 타임아웃이
나도 업로드는 실패하지 않고 텍스트 전용 추출로 성공 처리된다(레이아웃/표/이미지
인용은 없이). 이는 기존 OCR 실패 폴백과 동일한 사용자 경험이다.

### 3.6 설정 추가/변경

| 키 | 기본값 | 비고 |
|---|---|---|
| `mineru_base_url` | `"http://127.0.0.1:8100"` | MinerU 서비스 주소 |
| `mineru_timeout` | `300.0` | 초 단위. 대형 PDF(수백 페이지) 고려, 기존 `ocr_timeout`(이미지 1장당 120초)보다 훨씬 큰 문서 단위 타임아웃 |
| `mineru_enabled` | `True` | `False`면 3.5절 폴백과 동일하게 항상 `PyPDFLoader`만 사용 |
| `mineru_lang_list` | `["korean"]` | `/file_parse`의 `lang_list` 폼 필드로 그대로 전달. 서버 기본값(`["ch"]`, 중국어)은 이 앱의 한국어 문서에 맞지 않아 실측 후 추가(3.2절) |

**제거 대상** — PyMuPDF 레이아웃 판단 전용이라 MinerU 도입 후 무의미해지는
설정: `ocr_dpi`, `ocr_min_image_size`, `ocr_max_images_per_page`,
`ocr_page_text_threshold`, `ocr_row_tolerance`, `ocr_layout_order`.

**유지** — PaddleOCR 엔진 자체 설정(3.4절에서 계속 사용):
`ocr_enabled`(의미가 "이미지 블록 OCR 보강 여부"로 좁혀짐), `ocr_rec_model`,
`ocr_det_model`, `ocr_device`, `ocr_min_score`, `ocr_use_textline_orientation`,
`ocr_isolate_process`, `ocr_timeout`, `ocr_block_prefix`,
`ocr_keep_empty_placeholder`, `ocr_empty_placeholder`, `image_storage_dir`.

## 4. 범위

### 대상
- PDF 파싱(레이아웃/읽기순서/표/수식/이미지 인식)을 MinerU 경로로 완전 전환한다.
- 표/수식이 본문 텍스트로 검색 가능해진다(기존에는 표가 OCR로 뒤섞인 텍스트,
  수식은 인식조차 안 됐다).
- 도표 이미지 인용(원본 이미지를 출처 카드에 보여주는 기존 기능)은 동작을
  유지한다.

### 비대상 (YAGNI)
- docx/txt 처리 — MinerU 도입은 PDF 전용이다(기존 OCR 파이프라인도 PDF 전용).
- MinerU의 VLM 백엔드(수식/표 인식을 별도 비전-언어모델로 처리하는 옵션) —
  자체 호스팅 인프라 부담이 커 이 설계는 MinerU의 기본(pipeline) 백엔드를
  전제한다. 필요해지면 별도 설계로 다룬다.
- 청크-이미지 정밀 매핑 — 기존 설계(2026-08-28 문서)의 페이지 단위 근사를
  그대로 유지, 이 설계에서 다시 다루지 않는다.
- 여러 MinerU 인스턴스로의 부하 분산, 큐잉 — 이 저장소의 OCR 파이프라인은
  이미 업로드 요청 안에서 동기 실행되는 구조이고, 이 기능은 그 타이밍 특성을
  바꾸지 않는다.
- MinerU 서비스 자체의 헬스체크/자동 재시작 — 운영 관심사로 이 설계 범위 밖.

## 5. 한계

- MinerU 서비스가 이 프로젝트 배포에 새로운 외부 프로세스 의존성을 추가한다.
  기존에는 파이썬 패키지(paddlepaddle/paddleocr)를 서브프로세스로 격리하는
  정도였지만, 이제는 별도로 기동·관리해야 하는 상시 서비스가 하나 늘어난다.
- `table_body`/수식 LaTeX을 그대로 본문에 삽입하면 HTML/LaTeX 마크업 자체가
  임베딩 벡터에 노이즈로 섞여 들어갈 수 있다 — 3.3절에서 실측 후 정규화
  여부를 정하기 전까지는 잠정적 품질 리스크로 남는다.
- MinerU 실패 시 텍스트 전용 폴백은 표/수식/이미지 인용을 모두 포기한다 —
  기존 OCR 실패 폴백과 같은 수준의 성능 저하이지만, MinerU가 담당하는 범위가
  넓어진 만큼(레이아웃+표+수식+이미지) 폴백 시 잃는 것도 더 많아졌다.
