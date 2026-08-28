"""PaddleOCR wrapper used to turn document images into plain text.

The engine is loaded lazily so that importing this module never pulls in
paddle: only the first OCR call pays the model-loading cost.
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional
import logging
import multiprocessing
import threading

from app.config import get_settings
from app.services import ocr_worker

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class OcrLine:
    """A single recognised text line."""

    text: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1


def _polygon_to_box(polygon: Any) -> tuple[float, float, float, float]:
    """Convert a detection polygon (or box) into an (x0, y0, x1, y1) tuple."""
    points = [tuple(float(v) for v in point) for point in polygon]

    if len(points) == 2 and len(points[0]) == 1:
        # Already an axis aligned box expressed as [[x0, y0], [x1, y1]]
        (x0,), (y0,) = points
        return (x0, y0, x0, y0)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _box_from_flat(values: Any) -> tuple[float, float, float, float]:
    """Convert a flat [x0, y0, x1, y1] sequence into a box tuple."""
    x0, y0, x1, y1 = (float(v) for v in values)
    return (x0, y0, x1, y1)


def parse_ocr_result(raw: Any) -> list[OcrLine]:
    """Normalise PaddleOCR output into a list of :class:`OcrLine`.

    Supports both the PaddleOCR 3.x ``predict()`` payload (a list of dict-like
    results carrying ``rec_texts`` / ``rec_scores`` / ``rec_polys``) and the
    legacy 2.x ``ocr()`` payload (nested ``[[box, (text, score)], ...]`` lists).
    """
    if not raw:
        return []

    lines: list[OcrLine] = []

    for page in raw:
        if page is None:
            continue

        result = page
        # 3.x results expose the payload directly, but the JSON form nests it
        # one level deeper under "res".
        if isinstance(result, dict) and "rec_texts" not in result and "res" in result:
            result = result["res"]

        if isinstance(result, dict) and "rec_texts" in result:
            texts = result.get("rec_texts") or []
            scores = result.get("rec_scores") or []
            polys = result.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = result.get("dt_polys")
            boxes = result.get("rec_boxes")

            for idx, text in enumerate(texts):
                score = float(scores[idx]) if idx < len(scores) else 1.0
                box = (0.0, 0.0, 0.0, 0.0)
                if polys is not None and idx < len(polys):
                    box = _polygon_to_box(polys[idx])
                elif boxes is not None and idx < len(boxes):
                    box = _box_from_flat(boxes[idx])
                lines.append(OcrLine(text=str(text), score=score, box=box))
            continue

        # Legacy 2.x: page is a list of [polygon, (text, score)] entries.
        for entry in page:
            if not entry:
                continue
            polygon, recognition = entry[0], entry[1]
            text, score = recognition[0], recognition[1]
            lines.append(
                OcrLine(
                    text=str(text),
                    score=float(score),
                    box=_polygon_to_box(polygon),
                )
            )

    return lines


def group_rows(
    lines: list[OcrLine], overlap_ratio: float = 0.5
) -> list[list[OcrLine]]:
    """Group recognised lines into rows, top-to-bottom then left-to-right.

    Detection returns one box per text fragment, so a single visual line often
    arrives as several boxes.  Two boxes belong to the same row when they
    overlap vertically by at least ``overlap_ratio`` of the shorter one, which
    keeps the grouping independent of the render resolution.
    """
    if not lines:
        return []

    rows: list[list[OcrLine]] = []
    bounds: list[tuple[float, float]] = []

    for line in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        y0, y1 = line.box[1], line.box[3]
        if rows:
            top, bottom = bounds[-1]
            overlap = min(bottom, y1) - max(top, y0)
            shortest = min(bottom - top, y1 - y0)
            if shortest <= 0 or overlap >= overlap_ratio * shortest:
                rows[-1].append(line)
                bounds[-1] = (min(top, y0), max(bottom, y1))
                continue
        rows.append([line])
        bounds.append((y0, y1))

    return [sorted(row, key=lambda l: l.box[0]) for row in rows]


def lines_to_text(lines: list[OcrLine], min_score: float = 0.0) -> str:
    """Render recognised lines as reading-ordered text, one row per line."""
    kept = [l for l in lines if l.score >= min_score and l.text.strip()]
    rows = group_rows(kept)
    return "\n".join(
        " ".join(l.text.strip() for l in row) for row in rows
    ).strip()


class PaddleOCRService:
    """Thin, thread-safe wrapper around the PaddleOCR pipeline."""

    def __init__(
        self,
        rec_model_name: Optional[str] = None,
        det_model_name: Optional[str] = None,
        device: Optional[str] = None,
        min_score: Optional[float] = None,
        use_textline_orientation: Optional[bool] = None,
    ):
        self.rec_model_name = rec_model_name or settings.ocr_rec_model
        self.det_model_name = det_model_name or settings.ocr_det_model
        self.device = device or settings.ocr_device
        self.min_score = (
            settings.ocr_min_score if min_score is None else min_score
        )
        self.use_textline_orientation = (
            settings.ocr_use_textline_orientation
            if use_textline_orientation is None
            else use_textline_orientation
        )
        self._engine = None
        # Reentrant: read_lines() holds the lock while _predict() reaches
        # through the engine property, which locks again on first use.
        self._lock = threading.RLock()

    def _build_engine(self):
        from paddleocr import PaddleOCR

        logger.info(
            "Loading PaddleOCR (det=%s, rec=%s, device=%s)",
            self.det_model_name,
            self.rec_model_name,
            self.device,
        )
        return PaddleOCR(
            text_detection_model_name=self.det_model_name,
            text_recognition_model_name=self.rec_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=self.use_textline_orientation,
            device=self.device,
        )

    @property
    def engine(self):
        """The PaddleOCR pipeline, created on first use."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._build_engine()
        return self._engine

    def _predict(self, image: Any) -> Any:
        engine = self.engine
        if hasattr(engine, "predict"):
            return engine.predict(image)
        # PaddleOCR 2.x
        return engine.ocr(image, cls=self.use_textline_orientation)

    def read_lines(self, image: Any) -> list[OcrLine]:
        """Recognise an image (numpy BGR array or path) into text lines."""
        with self._lock:
            raw = self._predict(image)
        return parse_ocr_result(raw)

    def image_to_text(self, image: Any) -> str:
        """Recognise an image and return its text in reading order."""
        try:
            lines = self.read_lines(image)
        except Exception as e:  # pragma: no cover - depends on paddle runtime
            logger.error("OCR failed: %s", e)
            return ""
        return lines_to_text(lines, min_score=self.min_score)


class SubprocessOCRService:
    """Runs OCR in a dedicated process, and restarts it if it dies.

    PaddleOCR cannot share a process with torch (see :mod:`app.services.
    ocr_worker`), and this API process loads torch for the embedding model.
    One worker is kept alive across calls so the models load only once.
    """

    def __init__(
        self,
        target: Optional[Callable[[Any], str]] = None,
        timeout: Optional[float] = None,
    ):
        self._target = target or ocr_worker.recognise
        self.timeout = settings.ocr_timeout if timeout is None else timeout
        self._pool: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # "spawn", never "fork": a forked child would inherit the parent's
            # already-loaded torch, which is the thing being avoided.
            self._pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        return self._pool

    def _discard_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def image_to_text(self, image: Any) -> str:
        """Recognise an image in the worker process; "" if that is not possible."""
        with self._lock:
            for attempt in (1, 2):
                pool = self._ensure_pool()
                try:
                    return pool.submit(self._target, image).result(self.timeout)
                except BrokenProcessPool as e:
                    logger.error("OCR worker died (%s)", e)
                    self._discard_pool()  # a fresh worker for the next attempt
                except TimeoutError:
                    logger.error("OCR timed out after %ss", self.timeout)
                    self._discard_pool()
                    break
                except Exception as e:
                    logger.error("OCR failed: %s", e)
                    break
        return ""

    def shutdown(self) -> None:
        """Stop the worker process, if one is running."""
        with self._lock:
            self._discard_pool()


@lru_cache()
def get_ocr_service():
    """Cached OCR service so the models are only loaded once per worker."""
    if settings.ocr_isolate_process:
        return SubprocessOCRService()
    return PaddleOCRService()
