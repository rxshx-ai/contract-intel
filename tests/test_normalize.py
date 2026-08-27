"""Liberal in what we accept from the model, strict in what we surface."""

import pytest

from api.extract import (
    normalize_anchor,
    normalize_kind,
    normalize_owed_by,
    normalize_party,
    normalize_role,
)
from api.schemas import OurRole

BUYER = OurRole.BUYER
SELLER = OurRole.SELLER
US = "Contoso Systems Ltd."


@pytest.mark.parametrize("value,expected", [
    ("us", "us"), ("we", "us"), ("counterparty", "counterparty"),
    ("mutual", "mutual"), ("both parties", "mutual"), ("either party", "mutual"),
    ("na", "na"), ("", "na"), ("unknown", "na"),
])
def test_direct_vocabulary(value, expected):
    assert normalize_party(value, US, BUYER) == expected


def test_role_words_resolve_against_which_side_we_are_on():
    """The case that broke strict mode: the model writes the role, not our word."""
    assert normalize_party("Customer", US, BUYER) == "us"
    assert normalize_party("Provider", US, BUYER) == "counterparty"
    # Same words, other side of the paper, opposite answer.
    assert normalize_party("Customer", US, SELLER) == "counterparty"
    assert normalize_party("Provider", US, SELLER) == "us"


def test_named_entities_resolve_by_name():
    assert normalize_party("Contoso Systems Ltd.", US, BUYER) == "us"
    assert normalize_party("Contoso", US, BUYER) == "us"
    assert normalize_party("Northwind Observability, Inc.", US, BUYER) == "counterparty"


def test_mutual_nda_treats_role_words_as_mutual():
    assert normalize_party("Provider", US, OurRole.MUTUAL) == "mutual"


def test_owed_by_collapses_to_either():
    assert normalize_owed_by("mutual", US, BUYER) == "either"
    assert normalize_owed_by("", US, BUYER) == "either"
    assert normalize_owed_by("Customer", US, BUYER) == "us"


@pytest.mark.parametrize("value,expected", [
    ("notice", "notice"), ("NOTICE", "notice"), ("non-renewal", "notice"),
    ("auto-renewal", "renewal"), ("expiration", "expiry"), ("invoice", "payment"),
    ("reporting", "report"), ("cure period", "cure"),
])
def test_kind_aliases(value, expected):
    assert normalize_kind(value) == expected


def test_unmappable_kind_is_rejected_not_guessed():
    assert normalize_kind("arbitration_window") is None


def test_unknown_anchor_becomes_event_not_a_drop():
    """An obligation we cannot put on a calendar is still a real obligation.
    temporal.py reports it as conditional; dropping it would hide it."""
    assert normalize_anchor("the_third_tuesday") == "event"
    assert normalize_anchor("audit") == "event"
    assert normalize_anchor("") is None


@pytest.mark.parametrize("value,expected", [
    ("quarter_end", "quarter_end"), ("end of quarter", "quarter_end"),
    ("effective_date_anniversary", "anniversary"), ("annual", "anniversary"),
    ("end_of_month", "month_end"), ("breach_notice", "breach_date"),
])
def test_calendar_anchor_aliases(value, expected):
    assert normalize_anchor(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("term_end", "term_end"), ("end of term", "term_end"),
    ("renewal_date", "term_end"), ("commencement", "effective_date"),
    ("invoice", "invoice_date"), ("signing", "signature_date"),
])
def test_anchor_aliases(value, expected):
    assert normalize_anchor(value) == expected


def test_role_normalization():
    assert normalize_role("seller") == OurRole.SELLER
    assert normalize_role("Provider") == OurRole.SELLER
    assert normalize_role("mutual") == OurRole.MUTUAL
    assert normalize_role("anything else") == OurRole.BUYER
