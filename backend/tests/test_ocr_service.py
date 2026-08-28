"""PaddleOCR output normalisation, for both the 3.x and 2.x result shapes."""

import threading

from app.config import Settings
from app.services.ocr_service import (
    OcrLine,
    PaddleOCRService,
    lines_to_text,
    parse_ocr_result,
)


def test_defaults_use_the_korean_ppocrv5_recogniser():
    settings = Settings()

    assert settings.ocr_enabled is True
    assert settings.ocr_rec_model == "korean_PP-OCRv5_mobile_rec"
    assert PaddleOCRService().rec_model_name == "korean_PP-OCRv5_mobile_rec"


def test_image_storage_dir_defaults_alongside_other_storage_paths():
    settings = Settings()

    assert settings.image_storage_dir == "app/storage/images"


def test_engine_is_not_loaded_until_the_first_call():
    service = PaddleOCRService()

    assert service._engine is None


def test_parses_paddleocr_3x_results():
    raw = [
        {
            "rec_texts": ["매출 현황", "2024년"],
            "rec_scores": [0.99, 0.87],
            "rec_polys": [
                [[10, 10], [90, 10], [90, 30], [10, 30]],
                [[10, 40], [70, 40], [70, 60], [10, 60]],
            ],
        }
    ]

    lines = parse_ocr_result(raw)

    assert [l.text for l in lines] == ["매출 현황", "2024년"]
    assert lines[0].score == 0.99
    assert lines[0].box == (10.0, 10.0, 90.0, 30.0)


def test_parses_nested_json_style_3x_results():
    raw = [{"res": {"rec_texts": ["한글"], "rec_scores": [0.9], "dt_polys": [
        [[0, 0], [10, 0], [10, 10], [0, 10]]
    ]}}]

    assert [l.text for l in parse_ocr_result(raw)] == ["한글"]


def test_parses_legacy_2x_results():
    raw = [
        [
            [[[10, 40], [70, 40], [70, 60], [10, 60]], ("아래 줄", 0.91)],
            [[[10, 10], [90, 10], [90, 30], [10, 30]], ("위 줄", 0.95)],
        ]
    ]

    lines = parse_ocr_result(raw)

    assert {l.text for l in lines} == {"아래 줄", "위 줄"}
    assert lines_to_text(lines) == "위 줄\n아래 줄"


def test_empty_results_are_tolerated():
    assert parse_ocr_result(None) == []
    assert parse_ocr_result([None]) == []
    assert parse_ocr_result([{"rec_texts": [], "rec_scores": []}]) == []


def test_low_confidence_lines_are_dropped():
    lines = [
        OcrLine("확실", 0.95, (0, 0, 10, 10)),
        OcrLine("흐릿", 0.12, (0, 20, 10, 30)),
    ]

    assert lines_to_text(lines, min_score=0.5) == "확실"


def test_fragments_of_one_line_are_joined_left_to_right():
    """Detection splits a heading into boxes; they belong on one line."""
    lines = [
        OcrLine("현황", 0.9, (200, 12, 260, 40)),
        OcrLine("분기별", 0.9, (10, 10, 90, 38)),
        OcrLine("매출", 0.9, (100, 11, 170, 39)),
        OcrLine("2024년 3분기", 0.9, (10, 80, 200, 108)),
    ]

    assert lines_to_text(lines) == "분기별 매출 현황\n2024년 3분기"


def test_rows_are_separated_even_when_close_together():
    lines = [
        OcrLine("second", 0.9, (10, 32, 60, 58)),
        OcrLine("first", 0.9, (10, 4, 50, 30)),
    ]

    assert lines_to_text(lines) == "first\nsecond"


class _FakeEngine:
    def predict(self, image):
        return [{"rec_texts": ["인식 결과"], "rec_scores": [0.99], "rec_polys": [
            [[0, 0], [10, 0], [10, 10], [0, 10]]
        ]}]


def test_first_call_loads_the_engine_without_deadlocking():
    """read_lines() holds the lock while the engine property locks again."""

    class Service(PaddleOCRService):
        def _build_engine(self):
            return _FakeEngine()

    service = Service()
    result = {}

    def run():
        result["text"] = service.image_to_text(object())

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "image_to_text deadlocked on the engine lock"
    assert result["text"] == "인식 결과"
