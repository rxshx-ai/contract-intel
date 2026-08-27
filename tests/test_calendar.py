"""Calendar keeps three sources apart: system, computed, quoted."""

from datetime import date

import pytest

from api.calendar import build_calendar, by_month, in_range, literal_dates, summary

TODAY = date(2026, 8, 27)


@pytest.fixture(scope="module")
def bundles():
    from api import demo

    return demo.load(TODAY)


@pytest.fixture(scope="module")
def events(bundles):
    return build_calendar(bundles, TODAY)


def test_literal_dates_are_grounded(bundles):
    """Every date read out of the text keeps the span it came from."""
    for bundle in bundles:
        for doc in bundle.docs:
            for parsed, span in literal_dates(doc):
                assert doc.text[span.char_start:span.char_end] == span.quote
                assert str(parsed.year) in span.quote


def test_upload_dates_are_recorded(events):
    uploads = [e for e in events if e.kind == "uploaded"]
    assert len(uploads) == 6
    assert all(e.source == "system" for e in uploads)


def test_computed_deadlines_are_marked_actionable(events):
    computed = [e for e in events if e.source == "computed" and e.kind == "notice"]
    assert computed
    assert all(e.actionable for e in computed)


def test_dates_written_in_the_document_are_not_actionable(events):
    """A date in the text is not a deadline. Conflating them is the bug this
    product exists to fix."""
    quoted = [e for e in events if e.source == "quoted"]
    assert quoted
    assert all(not e.actionable for e in quoted)
    assert all(e.quote for e in quoted)


def test_the_derived_notice_deadline_is_present_and_absent_from_the_text(events):
    notice = [e for e in events
              if e.kind == "notice" and e.date == date(2026, 12, 31)]
    assert notice, "the derived renewal deadline must be on the calendar"
    assert notice[0].source == "computed"
    written = {e.date for e in events if e.source == "quoted"}
    assert date(2026, 12, 31) not in written   # it appears in no document


def test_events_are_sorted(events):
    assert [e.date for e in events] == sorted(e.date for e in events)


def test_range_filter(events):
    window = in_range(events, date(2026, 9, 1), date(2026, 12, 31))
    assert window
    assert all(date(2026, 9, 1) <= e.date <= date(2026, 12, 31) for e in window)


def test_month_grouping(events):
    months = by_month(events, TODAY)
    assert months
    assert sum(m["count"] for m in months) == len(events)
    assert all(m["label"] for m in months)


def test_summary_counts_sources_separately(events):
    s = summary(events, TODAY)
    assert s["documents"] == 6
    assert s["computed"] > s["written_in_documents"]
    assert s["total"] == len(events)
