"""Exhaustive graph-backed scans for Olympic event count and superlative questions."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

CATEGORY_RE = re.compile(
    r"\b(?:how many|which)\s+(?P<sport>.+?)\s+events?\s+at the\s+"
    r"(?P<year>\d{4})\s+(?P<season>summer|winter)\s+olympics\b",
    re.IGNORECASE,
)
THRESHOLD_RE = re.compile(
    r"\b(?P<operator>more than|greater than|at least|fewer than|less than)\s+(?P<value>\d+)\s+competitors\b",
    re.IGNORECASE,
)
TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)
FIELD_RE = re.compile(r"^\s*(?P<key>[\w]+):\s*(?P<value>.*?)\s*$", re.MULTILINE)
COMPETITOR_FIELD_RE = re.compile(r"^\s*competitors:\s*(?P<value>[\d,]+)\s*$", re.MULTILINE | re.IGNORECASE)
COMPETITOR_TEXT_RE = re.compile(
    r"\b(?P<value>\d[\d,]*|[a-z]+(?:[- ][a-z]+)?)\s+"
    r"(?P<noun>competitors|participants|cyclists|riders|athletes|sailors|fencers)\b",
    re.IGNORECASE,
)
SUPERLATIVE_RE = re.compile(r"\b(highest|most|largest|maximum|max)\b", re.IGNORECASE)
VENUE_DATE_RE = re.compile(
    r"\bevent held at\s+(?P<venue>.+?)\s+on\s+(?P<date>.+?)(?:\s+at the\s+\d{4}\s+(?:summer|winter)\s+olympics)?[?.]*$",
    re.IGNORECASE,
)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
DATE_TOKEN_RE = re.compile(
    r"(?:(?P<month1>january|february|march|april|may|june|july|august|september|october|november|december)\s+(?P<day1>\d{1,2})(?:\s*(?:-|–|—|to)\s*(?P<day2>\d{1,2}))?|"
    r"(?P<day3>\d{1,2})(?:\s*(?:-|–|—|to)\s*(?P<day4>\d{1,2}))?\s+(?P<month2>january|february|march|april|may|june|july|august|september|october|november|december))",
    re.IGNORECASE,
)
GRAPH_CACHE_SECONDS = 300
_GRAPH_EVENT_CACHE: dict[tuple[str, str], tuple[float, tuple[EventEvidence, ...], int, int]] = {}
_GRAPH_EVENT_CACHE_LOCK = threading.Lock()

_OPERATORS = {
    "more than": lambda count, threshold: count > threshold,
    "greater than": lambda count, threshold: count > threshold,
    "at least": lambda count, threshold: count >= threshold,
    "fewer than": lambda count, threshold: count < threshold,
    "less than": lambda count, threshold: count < threshold,
}
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}


@dataclass(frozen=True)
class CategoryQuestion:
    sport: str
    year: int
    season: str
    kind: Literal["count", "superlative"]
    threshold: int | None = None
    operator: str | None = None


@dataclass(frozen=True)
class EventEvidence:
    doc_id: str
    title: str
    competitors: int | None
    fields: dict[str, str]


@dataclass(frozen=True)
class CategoryScan:
    question: CategoryQuestion
    events: tuple[EventEvidence, ...]
    answer: str
    context: str
    latency_s: float
    complete: bool
    stop_reason: str

    @property
    def doc_ids(self) -> list[str]:
        return [event.doc_id for event in self.events]


@dataclass(frozen=True)
class VenueDateScan:
    events: tuple[EventEvidence, ...]
    answer: str
    context: str
    latency_s: float
    complete: bool
    stop_reason: str

    @property
    def doc_ids(self) -> list[str]:
        return [event.doc_id for event in self.events]


def parse_category_question(question: str) -> CategoryQuestion | None:
    match = CATEGORY_RE.search(question)
    if match is None:
        return None
    threshold = THRESHOLD_RE.search(question)
    if re.search(r"\bhow many\b", question, re.IGNORECASE) and threshold:
        operator = " ".join(threshold.group("operator").lower().split())
        return CategoryQuestion(
            sport=match.group("sport").strip(),
            year=int(match.group("year")),
            season=match.group("season").title(),
            kind="count",
            threshold=int(threshold.group("value")),
            operator=operator,
        )
    if match.group(0).lower().startswith("which") and SUPERLATIVE_RE.search(question):
        return CategoryQuestion(
            sport=match.group("sport").strip(),
            year=int(match.group("year")),
            season=match.group("season").title(),
            kind="superlative",
        )
    return None


def requires_event_scan(question: str) -> bool:
    return parse_category_question(question) is not None or _venue_date_query(question) is not None


def clear_graph_event_cache() -> None:
    with _GRAPH_EVENT_CACHE_LOCK:
        _GRAPH_EVENT_CACHE.clear()


def _date_pairs(value: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for match in DATE_TOKEN_RE.finditer(value.casefold()):
        month = MONTHS[match.group("month1") or match.group("month2")]
        first_day = int(match.group("day1") or match.group("day3"))
        last_day = match.group("day2") or match.group("day4")
        days = range(first_day, int(last_day) + 1) if last_day else (first_day,)
        pairs.update((month, day) for day in days)
    return pairs


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _venue_date_query(question: str) -> tuple[str, set[tuple[int, int]]] | None:
    match = VENUE_DATE_RE.search(question.strip())
    if match is None:
        return None
    venue = _normalized(match.group("venue"))
    dates = _date_pairs(match.group("date"))
    if not venue or not dates:
        return None
    return venue, dates


def _venue_date_candidates(question: str, events: tuple[EventEvidence, ...]) -> list[str] | None:
    parsed = _venue_date_query(question)
    if parsed is None:
        return None
    venue, dates = parsed
    return [
        event.doc_id
        for event in events
        if venue in _normalized(event.fields.get("venue", ""))
        and dates <= _date_pairs(event.fields.get("date", "") + " " + event.fields.get("dates", ""))
    ]


def _connection_key(connection: Any) -> tuple[str, str]:
    return str(getattr(connection, "host", "")), str(getattr(connection, "graphname", ""))


def _graph_event_records(connection: Any) -> tuple[tuple[EventEvidence, ...], int, int]:
    """Return parsed event content fetched from Savanna, plus graph and content counts."""
    key = _connection_key(connection)
    now = time.monotonic()
    with _GRAPH_EVENT_CACHE_LOCK:
        cached = _GRAPH_EVENT_CACHE.get(key)
        if cached and now - cached[0] < GRAPH_CACHE_SECONDS:
            return cached[1], cached[2], cached[3]

        document_count = int(connection.getVertexCount("Document"))
        contents = connection.getVertices("Content", select="id,text,ctype", limit=100000)
        document_contents = [
            row for row in contents
            if isinstance(row, dict) and row.get("attributes", {}).get("ctype") == "single"
        ]
        events = tuple(
            event
            for row in document_contents if isinstance(row, dict)
            if (event := _event_from_content(
                str(row.get("v_id", "")), str(row.get("attributes", {}).get("text", ""))
            )) is not None
            and event.fields.get("games")
        )
        value = (now, events, document_count, len(document_contents))
        _GRAPH_EVENT_CACHE[key] = value
        return events, document_count, len(document_contents)


def scan_venue_date(connection: Any, question: str) -> VenueDateScan | None:
    started = time.perf_counter()
    if _venue_date_query(question) is None:
        return None
    events, graph_documents, graph_contents = _graph_event_records(connection)
    candidates = _venue_date_candidates(question, events)
    if candidates is None:
        return None
    if len(candidates) != 1:
        return None
    event = next(event for event in events if event.doc_id == candidates[0])
    if graph_contents < graph_documents or not event.fields.get("gold"):
        reason = "graph inventory incomplete: missing Document Content or gold-medal field"
        return VenueDateScan((event,), "", _format_event(event), time.perf_counter() - started, False, reason)
    answer = f"Final answer: {event.fields['gold']} [{event.doc_id}]"
    return VenueDateScan(
        (event,),
        answer,
        _format_event(event),
        time.perf_counter() - started,
        True,
        f"unique graph event matched venue and date: {event.title}",
    )


def _number_value(value: str) -> int | None:
    value = value.strip().casefold().replace(",", "")
    value = re.sub(r"^the\s+", "", value)
    if value.isdigit():
        return int(value)
    value = value.replace("-", " ")
    parts = value.split()
    if not parts or any(part not in _NUMBER_WORDS for part in parts):
        return None
    total = sum(_NUMBER_WORDS[part] for part in parts)
    return total if total > 0 else None


def _decode_graph_text(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\t", "\t")


def _infobox_text(text: str) -> str:
    _, marker, remaining = text.partition("[Infobox Olympic event]")
    return remaining.split("\n\n", 1)[0] if marker else ""


def _event_from_content(doc_id: str, text: str) -> EventEvidence | None:
    text = _decode_graph_text(text)
    title_match = TITLE_RE.search(text)
    if title_match is None:
        return None
    infobox = _infobox_text(text)
    fields = {
        match.group("key").casefold(): match.group("value").strip()
        for match in FIELD_RE.finditer(infobox)
    }
    competitors_match = COMPETITOR_FIELD_RE.search(infobox)
    competitors = _number_value(competitors_match.group("value")) if competitors_match else None
    if competitors is None:
        for match in COMPETITOR_TEXT_RE.finditer(text):
            competitors = _number_value(match.group("value"))
            if competitors is not None:
                break
    return EventEvidence(doc_id.upper(), title_match.group(1).strip(), competitors, fields)


def _format_event(event: EventEvidence) -> str:
    parts = [f"[{event.doc_id}] {event.title}"]
    if event.competitors is not None:
        parts.append(f"competitors: {event.competitors}")
    for key in ("venue", "date", "dates", "gold"):
        if event.fields.get(key):
            parts.append(f"{key}: {event.fields[key]}")
    return " | ".join(parts)


def _calculated_answer(parsed: CategoryQuestion, events: tuple[EventEvidence, ...]) -> tuple[str, str]:
    if parsed.kind == "count":
        comparator = _OPERATORS[parsed.operator or "more than"]
        matches = [event for event in events if comparator(event.competitors or 0, parsed.threshold or 0)]
        return (
            str(len(matches)),
            f"complete graph category: compared {len(events)} events; counted {len(matches)} meeting the threshold",
        )
    winner = max(events, key=lambda event: event.competitors or 0)
    return winner.title, f"complete graph category: ranked {len(events)} events by competitor count"


def scan_category(connection: Any, question: str) -> CategoryScan | None:
    parsed = parse_category_question(question)
    if parsed is None:
        return None
    started = time.perf_counter()
    all_events, graph_documents, graph_contents = _graph_event_records(connection)
    prefix = _normalized(f"{parsed.sport} at the {parsed.year} {parsed.season} Olympics")
    events = tuple(
        event for event in all_events
        if _normalized(event.title).startswith(prefix)
        and _normalized(event.fields.get("games", "")) == _normalized(f"{parsed.year} {parsed.season}")
    )
    complete = (
        bool(events)
        and graph_contents >= graph_documents
        and all(event.competitors is not None for event in events)
    )
    context = "\n".join(_format_event(event) for event in events)
    if not complete:
        missing_counts = sum(event.competitors is None for event in events)
        reason = (
            f"graph category scan incomplete: {len(events)} matching event(s), "
            f"{graph_documents} documents, {graph_contents} contents, "
            f"{missing_counts} event(s) missing competitor counts"
        )
        return CategoryScan(parsed, events, "", context, time.perf_counter() - started, False, reason)

    answer_value, stop_reason = _calculated_answer(parsed, events)
    citations = ", ".join(event.doc_id for event in events) if parsed.kind == "count" else next(
        event.doc_id for event in events if event.title == answer_value
    )
    answer = f"Final answer: {answer_value} [{citations}]"
    return CategoryScan(parsed, events, answer, context, time.perf_counter() - started, True, stop_reason)
