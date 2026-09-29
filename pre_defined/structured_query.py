"""Deterministic answers for aggregation/superlative questions.

Chunk-similarity retrieval with a small top_k can never see every document in
a category, so "how many X events had more than N competitors" and "which X
event had the most competitors" are structurally unanswerable by asking an LLM
to eyeball a handful of retrieved chunks. Both question types reference facts
that are already present as an "[Infobox Olympic event]" block in each
document's text (event, games, competitors, ...), so we extract those fields
once into a local index and answer the question with a direct filter/count
instead of routing it through GraphRAG at all.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from pre_defined.main import CORPUS_PATH


STRUCTURED_QTYPES = {"aggregation", "superlative"}

_GAMES_RE = re.compile(r"^\s*games:\s*(?P<year>\d{4})\s+(?P<season>Summer|Winter)\s*$", re.IGNORECASE | re.MULTILINE)
_COMPETITORS_RE = re.compile(r"^\s*competitors:\s*(?P<count>\d+)\s*$", re.IGNORECASE | re.MULTILINE)

_QUESTION_RE = re.compile(
    r"(?:how many|which)\s+(?P<sport>.+?)\s+events?\s+at the\s+(?P<year>\d{4})\s+(?P<season>summer|winter)\s+olympics",
    re.IGNORECASE,
)
_THRESHOLD_RE = re.compile(
    r"(more than|at least|fewer than|less than)\s+(\d+)\s+competitors",
    re.IGNORECASE,
)
_OPERATORS = {
    "more than": lambda value, threshold: value > threshold,
    "at least": lambda value, threshold: value >= threshold,
    "fewer than": lambda value, threshold: value < threshold,
    "less than": lambda value, threshold: value < threshold,
}


@dataclass(frozen=True)
class Event:
    doc_id: str
    title: str
    sport: str
    games: str
    competitors: int | None


def _iter_corpus(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if line.strip():
                yield json.loads(line)


def _parse_event(document: dict) -> Event | None:
    title = document.get("title", "")
    if " at the " not in title:
        return None
    sport = title.split(" at the ", 1)[0].strip()
    text = document.get("text", "")
    games_match = _GAMES_RE.search(text)
    if not games_match:
        return None
    competitors_match = _COMPETITORS_RE.search(text)
    return Event(
        doc_id=document["doc_id"],
        title=title,
        sport=sport,
        games=f"{games_match['year']} {games_match['season'].title()}",
        competitors=int(competitors_match["count"]) if competitors_match else None,
    )


@lru_cache(maxsize=1)
def load_event_index(corpus_path: Path = CORPUS_PATH) -> tuple[Event, ...]:
    return tuple(event for document in _iter_corpus(corpus_path) if (event := _parse_event(document)) is not None)


def _matching_events(sport: str, games: str, index: tuple[Event, ...]) -> list[Event]:
    return [event for event in index if event.sport.lower() == sport.lower() and event.games == games]


def _parse_category(question: str) -> tuple[str, str] | None:
    match = _QUESTION_RE.search(question)
    if not match:
        return None
    games = f"{match['year']} {match['season'].title()}"
    return match["sport"].strip(), games


def answer_aggregation(question: str, index: tuple[Event, ...] | None = None) -> str | None:
    category = _parse_category(question)
    threshold_match = _THRESHOLD_RE.search(question)
    if category is None or threshold_match is None:
        return None
    sport, games = category
    comparator = _OPERATORS[threshold_match.group(1).lower()]
    threshold = int(threshold_match.group(2))
    events = _matching_events(sport, games, index if index is not None else load_event_index())
    matching = [event for event in events if event.competitors is not None and comparator(event.competitors, threshold)]
    if not events:
        return None
    return str(len(matching))


def answer_superlative(question: str, index: tuple[Event, ...] | None = None) -> str | None:
    category = _parse_category(question)
    if category is None:
        return None
    sport, games = category
    events = _matching_events(sport, games, index if index is not None else load_event_index())
    ranked = [event for event in events if event.competitors is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda event: event.competitors).title


def try_structured_answer(question: str, qtype: str) -> str | None:
    """Return a deterministic answer for aggregation/superlative qtypes, else None."""
    if qtype == "aggregation":
        return answer_aggregation(question)
    if qtype == "superlative":
        return answer_superlative(question)
    return None
