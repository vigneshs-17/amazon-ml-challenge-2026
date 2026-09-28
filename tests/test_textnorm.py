"""Tests for Unicode normalization, tokenization, placeholder handling, and script preservation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from textnorm import normalize_basic, tokens


def test_unicode_accents_normalization():
    """Verify NFKD decomposition and accent stripping for French & multilingual Latin text."""
    raw = "Société Générale de Banque - Café & Restaurant"
    norm = normalize_basic(raw)
    assert "societe" in norm
    assert "generale" in norm
    assert "cafe" in norm
    assert "restaurant" in norm
    assert "&" not in norm


def test_devanagari_combining_marks_preserved():
    """Verify Devanagari characters and vowel combining marks are never destroyed."""
    raw = "राम मार्केटिंग प्राइवेट लिमिटेड"
    norm = normalize_basic(raw)
    assert norm == "राम मार्केटिंग प्राइवेट लिमिटेड"

    kannada = "ಬೆಂಗಳೂರು"
    assert normalize_basic(kannada) == "ಬೆಂಗಳೂರು"


def test_placeholder_tag_and_null_handling():
    """Verify <NULL> and <CITY_NAME> tags are handled while bare 'Null' address tokens are preserved."""
    # Whole-field <NULL> placeholder
    assert normalize_basic("<NULL>") == ""
    assert normalize_basic("<null>") == ""
    assert normalize_basic("") == ""
    assert normalize_basic(None) == ""

    # Embedded <NULL> in address
    assert normalize_basic("2769 Carefree Cir, <NULL>, Flagstaff, Arizona") == "2769 carefree cir flagstaff arizona"

    # Corrupted synthetic city tags
    assert normalize_basic("<CITY_NAME> Chit Limited") == "chit limited"
    assert normalize_basic("<ClTY_NAME> Chit Limited") == "chit limited"
    assert normalize_basic("<CÍTY_NAME> Chit Limited") == "chit limited"

    # Bare, unbracketed "null" is REAL address text in India (e.g. Null Bazaar) and must be preserved
    assert normalize_basic("Null Bazaar, Mandvi, Mumbai") == "null bazaar mandvi mumbai"
    assert normalize_basic("North Nullivilai, Kalkulam") == "north nullivilai kalkulam"


def test_tokenization_and_whitespace():
    """Verify tokenization splits words cleanly and drops extraneous punctuation and spaces."""
    text = "   Reliance   Industries    Limited,   Mumbai.  "
    t_list = tokens(text)
    assert t_list == ["reliance", "industries", "limited", "mumbai"]
