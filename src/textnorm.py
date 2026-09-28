"""Country-agnostic text normalization + script detection.

Only algorithmic Unicode operations (stdlib `unicodedata`), no external
dictionaries. Accent stripping is restricted to Latin base characters so that
Devanagari (and any other script) vowel signs are never destroyed.
"""

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
# Explicit missing-value placeholder observed in S2/S3 (e.g. address "<NULL>").
NULL_PLACEHOLDERS = {"<null>"}

# Bracket-delimited template/placeholder tags, matched embedded anywhere in a
# field (not just whole-field), e.g. "2769 Carefree Cir, <NULL>, Flagstaff,
# Arizona" or "<CITY_NAME> Chit Limited". This is a SMALL, evidence-grounded
# list -- every entry below was actually observed via grep/audit on the raw
# files (train/test source2/3), not guessed: <city_name> is the clean form,
# the rest are its corrupted siblings found co-occurring with it in the same
# files (single-character substitutions/transpositions, consistent with
# synthetic noise injection into one template string). Matched with a plain
# substring/regex approach (not the earlier draft's broad "any bracketed
# lowercase word" pattern), because a broad pattern proved unsafe -- it also
# matched decorative junk-punctuation names like "<< Elisabeth >" (a real
# name wrapped in stray "<"/">" characters, the same noise family as
# "#4 Presidio" or "[INCORPORATED] Peak..."), which must NOT be stripped.
# Applied on the casefold+accent-stripped string, so the accented sibling
# <cíty_name> normalizes to <city_name> and is covered by that one entry.
_CITY_NAME_TAG_RE = re.compile(
    "<(?:city_name|clty_name|cit_name|city_iname|cbi_ynome|fity_name|" "icity_nem|ccaty_nae)>"
)
# <NULL>, embedded anywhere in a field, not just when it is the whole field
# (the whole-field case is handled by NULL_PLACEHOLDERS below; most
# occurrences -- e.g. "067 Production Ct, <NULL>, Independence, KY" -- are
# one comma-separated component among several).
_NULL_TAG_RE = re.compile(r"<null>")
#
# Deliberately NOT stripped: a bare, unbracketed "NULL"/"null" token. Audit
# evidence shows this is frequently a REAL address fragment in India records
# -- "Null Bazaar" (a genuine Mumbai locality), "Nullivilai", "Nullipady",
# "Nullikadu" (genuine Tamil Nadu/Kerala place-name fragments) all appear in
# train_source1.tsv. Blindly removing a standalone "null" token would
# corrupt real addresses, so only the unambiguous bracketed form is treated
# as a placeholder.

# Map every Unicode punctuation (P*) / symbol (S*) code point to a space.
# Category-based (not regex \w) so combining marks of Indic scripts survive.
_PUNCT_TABLE = {cp: " " for cp in range(0x110000) if unicodedata.category(chr(cp))[0] in "PS"}

# Script regexes, used for vectorised flags. Written with literal characters
# (non-raw strings) so they compile under both Python `re` and pyarrow's RE2.
SCRIPT_PATTERNS = {
    "latin_basic": "[A-Za-z]",
    "latin_accented": "[À-ɏ]",
    "devanagari": "[ऀ-ॿ]",
    "other_indic": "[ঀ-෿]",  # Bengali..Sinhala (Tamil, Telugu, ...)
    "cjk": "[぀-ヿ一-鿿]",
    "arabic": "[؀-ۿ]",
    "cyrillic": "[Ѐ-ӿ]",
    "digit": "[0-9]",
    "non_ascii": "[^\u0001-\u007f]",
}


def strip_latin_accents(s: str) -> str:
    out = []
    prev_latin = False
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(ch):
            if prev_latin:
                continue  # drop accent on a Latin base char
            out.append(ch)
            continue
        prev_latin = ch < "ɐ"
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_basic(s: str) -> str:
    """NFKC, casefold, Latin-accent strip, punctuation->space, collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    if s.strip() in NULL_PLACEHOLDERS:
        return ""
    s = strip_latin_accents(s)
    s = _CITY_NAME_TAG_RE.sub(" ", s)
    s = _NULL_TAG_RE.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", s).strip()


def tokens(s: str):
    return normalize_basic(s).split()


def _self_test():
    cases = [
        # (input, expected output)
        ("2769 Carefree Cir, <NULL>, Flagstaff, Arizona", "2769 carefree cir flagstaff arizona"),
        ("<CITY_NAME> Chit Limited", "chit limited"),
        ("<ClTY_NAME> Chit Limited", "chit limited"),
        ("<CIT_NAME> Chit Limited", "chit limited"),
        ("<CITY_iNAME> Chit Limited", "chit limited"),
        ("<CBI_YNOME> Chit Limited", "chit limited"),
        ("<CÍTY_NAME> Chit Limited", "chit limited"),
        # decorative junk-bracket noise around a REAL name must NOT be
        # treated as a placeholder tag -- only its "<"/">" punctuation is
        # stripped (by the ordinary punctuation step), the name survives
        ("<< Elisabeth >", "elisabeth"),
        ("<NULL>", ""),  # whole-field placeholder, unchanged behaviour
        # bracketed content WITH digits/punctuation must survive as tokens,
        # not be swallowed by the tag patterns (which only match the
        # specific known letter-only tag spellings above)
        ("Plot <1-98/9/3/31> Sector 5", "plot 1 98 9 3 31 sector 5"),
        # bare, unbracketed "null"/"NULL" is real address text here and
        # must NOT be stripped
        (
            "House No. 641/486 Main Road K S Nagar Null, Berhampur",
            "house no 641 486 main road k s nagar null berhampur",
        ),
        ("Null Bazaar, Mandvi, Mumbai", "null bazaar mandvi mumbai"),
        ("North Nullivilai, Kalkulam", "north nullivilai kalkulam"),
        ("067 PRODUCTION CT, NULL, INDEPENDENCE, KY", "067 production ct null independence ky"),
        # Devanagari / other scripts must survive untouched
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड"),
        ("ಬೆಂಗಳೂರು", "ಬೆಂಗಳೂರು"),
        # French accents + legal suffix punctuation
        ("Dauphine & Cie Développement S.A.S.", "dauphine cie developpement s a s"),
        ("N°24 R DE LA TRANQUILITE", "n 24 r de la tranquilite"),
    ]
    for inp, expected in cases:
        got = normalize_basic(inp)
        assert got == expected, f"normalize_basic({inp!r}) = {got!r}, expected {expected!r}"
    print(f"textnorm self-test OK ({len(cases)}/{len(cases)} cases)")


if __name__ == "__main__":
    _self_test()
