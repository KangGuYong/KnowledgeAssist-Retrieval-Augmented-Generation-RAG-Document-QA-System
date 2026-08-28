"""The PDF loader must replace image regions with their OCR text, in place."""

import io

import fitz
import pytest

from app.services import pdf_ocr
from app.services.pdf_ocr import extract_pages


class StubOCR:
    """Stands in for PaddleOCR: records what it was asked to read."""

    def __init__(self, text="스캔된 표 내용"):
        self.text = text
        self.calls = []

    def image_to_text(self, image):
        self.calls.append(image)
        return self.text


def _image_bytes(color=(220, 40, 40), size=(120, 90)):
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, *size))
    pixmap.set_rect(pixmap.irect, color)
    return pixmap.tobytes("png")


def _build_pdf(tmp_path, name="sample.pdf", with_text=True, image_rect=(72, 200, 300, 380)):
    doc = fitz.open()
    page = doc.new_page()
    if with_text:
        page.insert_text((72, 100), "Heading above the image", fontsize=14)
        page.insert_text((72, 500), "Paragraph below the image", fontsize=14)
    page.insert_image(fitz.Rect(*image_rect), stream=_image_bytes())
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return str(path)


def test_image_region_is_replaced_by_ocr_text_in_reading_order(tmp_path):
    ocr = StubOCR("표: 매출 120억")
    pages = extract_pages(_build_pdf(tmp_path), ocr=ocr)

    assert len(pages) == 1
    page = pages[0]
    text = page.text

    assert "표: 매출 120억" in text
    assert page.ocr_image_count == 1
    assert page.full_page_ocr is False
    assert page.page_number == 1

    # The recognised text sits where the image sat: after the heading, before
    # the paragraph that follows it.
    assert text.index("Heading") < text.index("표: 매출") < text.index("Paragraph")


def test_ocr_text_is_labelled_so_chunks_show_their_origin(tmp_path):
    pages = extract_pages(_build_pdf(tmp_path), ocr=StubOCR("인식된 문장"))

    assert "[이미지 텍스트]\n인식된 문장" in pages[0].text


def test_only_the_image_region_is_rendered_for_ocr(tmp_path):
    ocr = StubOCR()
    # 240 x 180 pt keeps the 120 x 90 image's aspect ratio, so the placed image
    # fills the rectangle exactly.
    extract_pages(_build_pdf(tmp_path, image_rect=(72, 200, 312, 380)), ocr=ocr)

    assert len(ocr.calls) == 1
    height, width = ocr.calls[0].shape[:2]
    # The image region only, rendered at the default 200 dpi.
    assert width == pytest.approx(240 / 72 * 200, abs=3)
    assert height == pytest.approx(180 / 72 * 200, abs=3)


def test_scanned_page_falls_back_to_one_full_page_pass(tmp_path):
    ocr = StubOCR("스캔 페이지 전문")
    path = _build_pdf(tmp_path, with_text=False, image_rect=(0, 0, 400, 600))

    pages = extract_pages(path, ocr=ocr)

    assert len(ocr.calls) == 1
    assert pages[0].full_page_ocr is True
    assert "스캔 페이지 전문" in pages[0].text


def test_many_image_fragments_are_read_in_a_single_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_ocr.settings, "ocr_max_images_per_page", 2)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Some real text on the page that is long enough")
    for idx in range(4):
        page.insert_image(
            fitz.Rect(72, 100 + idx * 100, 272, 180 + idx * 100),
            stream=_image_bytes(color=(idx * 40, 10, 10)),
        )
    path = tmp_path / "fragments.pdf"
    doc.save(path)
    doc.close()

    ocr = StubOCR("한 번에 읽은 페이지")
    pages = extract_pages(str(path), ocr=ocr)

    assert len(ocr.calls) == 1
    assert pages[0].full_page_ocr is True


def test_repeated_images_are_recognised_once(tmp_path):
    doc = fitz.open()
    stream = _image_bytes()
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 60), "Header text that is long enough to keep")
        page.insert_image(fitz.Rect(72, 200, 300, 380), stream=stream)
    path = tmp_path / "logo.pdf"
    doc.save(path)
    doc.close()

    ocr = StubOCR("로고 문구")
    pages = extract_pages(str(path), ocr=ocr)

    assert len(pages) == 3
    assert all("로고 문구" in p.text for p in pages)
    assert len(ocr.calls) == 1  # cached across pages by image content


def test_tiny_images_are_skipped(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Text long enough to avoid the scanned-page path")
    page.insert_image(fitz.Rect(72, 100, 92, 120), stream=_image_bytes(size=(20, 20)))
    path = tmp_path / "icon.pdf"
    doc.save(path)
    doc.close()

    ocr = StubOCR()
    pages = extract_pages(str(path), ocr=ocr)

    assert ocr.calls == []
    assert pages[0].ocr_image_count == 0


class FailingOCR:
    def image_to_text(self, image):
        raise RuntimeError("paddle exploded")


def test_unreadable_image_leaves_the_rest_of_the_page_intact(tmp_path):
    pages = extract_pages(_build_pdf(tmp_path), ocr=FailingOCR())

    assert "Heading above the image" in pages[0].text
    assert pages[0].ocr_image_count == 0


def test_unreadable_scanned_page_does_not_abort_the_document(tmp_path):
    path = _build_pdf(tmp_path, with_text=False, image_rect=(0, 0, 400, 600))

    pages = extract_pages(path, ocr=FailingOCR())

    assert len(pages) == 1
    assert pages[0].full_page_ocr is False
    assert pages[0].text == ""


def test_image_block_is_saved_as_png(tmp_path):
    """블록 단위 이미지는 image_dir에 PNG로 저장되고 page.image_ids에 실린다."""
    image_dir = tmp_path / "images"
    pdf_path = _build_pdf(tmp_path)

    pages = extract_pages(pdf_path, ocr=StubOCR(), image_dir=image_dir)

    assert len(pages[0].image_ids) == 1
    image_id = pages[0].image_ids[0]
    saved = image_dir / f"{image_id}.png"
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_no_image_dir_skips_saving_without_failing(tmp_path):
    """image_dir=None이면(기존 호출부 호환) 저장을 건너뛰고 텍스트 추출은 그대로 동작한다."""
    pdf_path = _build_pdf(tmp_path)

    pages = extract_pages(pdf_path, ocr=StubOCR())

    assert "스캔된 표 내용" in pages[0].text
    assert pages[0].image_ids == []


def test_repeated_image_across_pages_is_saved_once(tmp_path):
    """같은 이미지 바이트가 여러 페이지에 반복돼도 같은 image_id, 같은 파일로 수렴한다."""
    doc = fitz.open()
    image_bytes = _image_bytes()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 60), "Header text that is long enough to keep")
        page.insert_image(fitz.Rect(72, 200, 300, 380), stream=image_bytes)
    pdf_path = tmp_path / "repeated.pdf"
    doc.save(pdf_path)
    doc.close()
    image_dir = tmp_path / "images"

    pages = extract_pages(str(pdf_path), ocr=StubOCR(), image_dir=image_dir)

    assert pages[0].image_ids == pages[1].image_ids
    assert len(list(image_dir.glob("*.png"))) == 1


def test_image_save_failure_does_not_break_text_extraction(tmp_path):
    """저장 디렉터리를 쓸 수 없어도 OCR 텍스트 추출은 실패하지 않는다."""
    pdf_path = _build_pdf(tmp_path)
    unwritable = tmp_path / "not_a_directory"
    unwritable.write_text("occupied")  # 디렉터리로 만들 수 없는 경로

    pages = extract_pages(pdf_path, ocr=StubOCR(), image_dir=unwritable)

    assert "스캔된 표 내용" in pages[0].text
    assert pages[0].image_ids == []


def test_full_page_scan_is_saved_with_a_page_scoped_id(tmp_path):
    """전체 페이지 OCR(스캔본/이미지 과다)도 하나의 이미지로 저장된다."""
    path = _build_pdf(tmp_path, with_text=False, image_rect=(0, 0, 400, 600))
    image_dir = tmp_path / "images"

    pages = extract_pages(path, ocr=StubOCR(), image_dir=image_dir)

    assert pages[0].full_page_ocr is True
    assert pages[0].image_ids == ["p1_full"]
    assert (image_dir / "p1_full.png").exists()
