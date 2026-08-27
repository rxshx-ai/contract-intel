"""The PDF path, including text hidden from a human reader."""

import pathlib

import pytest

from api.firewall import inspect, scan_pdf_layers
from api.ingest import find_span, ingest_pdf

PDF = pathlib.Path("contracts/poisoned_scan.pdf")

pytestmark = pytest.mark.skipif(not PDF.exists(),
                                reason="run eval/make_poisoned_pdf.py first")


@pytest.fixture(scope="module")
def scanned():
    return ingest_pdf(str(PDF))


def test_offsets_survive_the_pdf_path(scanned):
    """Invariant 1 must hold for real PDF extraction, not just plain text."""
    for line in scanned.text.split("\n"):
        line = line.strip()
        if len(line) < 25:
            continue
        span = find_span(scanned, line)
        assert span is not None
        assert span.is_grounded_in(scanned.text)


def test_page_marks_cover_the_document(scanned):
    assert scanned.pages
    for mark in scanned.pages:
        assert scanned.text[mark.char_start:mark.char_end].strip()


def test_all_four_hidden_channels_are_detected(scanned):
    kinds = {i.kind for i in scan_pdf_layers(str(PDF))}
    assert kinds == {"invisible_text", "tiny_font", "offscreen_text", "metadata_payload"}


def test_scanned_document_is_quarantined(scanned):
    report = inspect(scanned, str(PDF))
    assert report.quarantined
    assert len(report.indicators) >= 10


def test_visible_text_is_still_extracted(scanned):
    assert "MASTER SERVICES AGREEMENT" in scanned.text
    assert "one hundred twenty (120)" in scanned.text
