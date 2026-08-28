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
