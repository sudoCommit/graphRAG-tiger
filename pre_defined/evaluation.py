"""Evaluation utilities for the pre-defined GraphRAG corpus.

Scoring is qtype-aware since the gold answers in the eval set take two very
different shapes:
- aggregation / lookup: a number (e.g. "5", "26").
- superlative / temporal / multi_hop / other: a name or title (e.g. "Chen Ding",
  "Athletics at the 2008 Summer Olympics - Men's marathon").

A single string-similarity metric (token F1, strict exact match) fits neither
case well: it either can't tell "5" from "50" apart, or it heavily penalizes a
correct name/number wrapped in an explanatory sentence. `score_answer` picks
the right comparison per qtype and falls back to fuzzy partial matching for
near-misses (accents, punctuation, alternate dash characters, etc.).
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


QUESTIONS_ROOT = Path(__file__).resolve().parent / "data" / "questions"

# aggregation/lookup answers are counts; everything else is a name or title.
NUMERIC_QTYPES = {"aggregation", "lookup"}
FUZZY_MATCH_THRESHOLD = 0.85

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def load_questions(dataset: str) -> list[dict[str, Any]]:
    path = QUESTIONS_ROOT / f"{dataset}.jsonl"
    with path.open(encoding="utf-8") as questions_file:
        return [json.loads(line) for line in questions_file if line.strip()][:10]


def normalize_answer(value: str) -> str:
    """Lowercase, strip accents and dash/punctuation variants for robust comparison."""
    value = value.replace("\u2013", "-").replace("\u2014", "-")  # en dash / em dash -> hyphen
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9 ]", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def _extract_numbers(text: str) -> list[int]:
    numbers = [int(match) for match in re.findall(r"\d+", text)]
    numbers.extend(_NUMBER_WORDS[word] for word in normalize_answer(text).split() if word in _NUMBER_WORDS)
    return numbers


def _partial_ratio(short: str, long: str) -> float:
    """Best similarity of `short` against any equal-length window inside `long`."""
    if not short or not long:
        return 0.0
    if len(short) > len(long):
        short, long = long, short
    best = 0.0
    for block in SequenceMatcher(None, short, long).get_matching_blocks():
        start = max(block.b - block.a, 0)
        window = long[start:start + len(short)]
        best = max(best, SequenceMatcher(None, short, window).ratio())
    return best


def _score_numeric(prediction: str, expected_answers: list[str]) -> dict[str, Any]:
    predicted_numbers = _extract_numbers(prediction)
    for expected in expected_answers:
        if any(number in predicted_numbers for number in _extract_numbers(expected)):
            return {"correct": True, "method": "numeric", "confidence": 1.0}
    return {"correct": False, "method": "numeric", "confidence": 0.0}


def _score_text(prediction: str, expected_answers: list[str]) -> dict[str, Any]:
    normalized_prediction = normalize_answer(prediction)
    best_confidence = 0.0
    for expected in expected_answers:
        normalized_expected = normalize_answer(expected)
        if not normalized_expected:
            continue
        if normalized_expected in normalized_prediction:
            return {"correct": True, "method": "contains", "confidence": 1.0}
        best_confidence = max(best_confidence, _partial_ratio(normalized_expected, normalized_prediction))
    return {
        "correct": best_confidence >= FUZZY_MATCH_THRESHOLD,
        "method": "fuzzy",
        "confidence": round(best_confidence, 3),
    }


def score_answer(prediction: str, expected_answers: list[str], qtype: str) -> dict[str, Any]:
    """Qtype-aware scoring. Returns {'correct': bool | None, 'method': str, 'confidence': float}."""
    if not expected_answers or not prediction:
        return {"correct": None, "method": "n/a", "confidence": 0.0}
    if qtype in NUMERIC_QTYPES:
        return _score_numeric(prediction, expected_answers)
    return _score_text(prediction, expected_answers)