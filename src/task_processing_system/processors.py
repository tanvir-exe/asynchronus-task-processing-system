"""Pure processing functions for the two task types.

These functions have no I/O and no knowledge of FastAPI or the database,
so they are easy to unit test. They are CPU-bound; the worker runs them in
a thread so they never block the event loop.

Design decisions (also documented in the README):

Text analysis
  * A "word" is a run of letters/digits, optionally joined by an apostrophe
    or hyphen inside the word ("don't", "well-known"). Decimal numbers such
    as "3.14" count as one word. Matching is
    case-insensitive, so "The" and "the" are the same word.
  * Average word length counts the characters of each word token.
  * Sentences end at '.', '!' or '?' followed by whitespace (or end of text).
    The longest sentence is the one with the most words; ties go to the first.
    Known limitation: abbreviations such as "Mr. Smith" are split.

Transaction analysis
  * A record is valid only if it has a non-empty transaction_id and
    customer_id, a finite numeric amount greater than zero (booleans and
    numeric strings are rejected), and an ISO 8601 timestamp.
  * Duplicates are valid records that share a transaction_id. The first
    occurrence is kept; later copies are reported as duplicates and are
    excluded from revenue and customer totals.
  * Invalid records are reported with their index and the reasons.
  * Money is summed with Decimal to avoid float rounding drift.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Any

# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]+(?:['\u2019\-][^\W_]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def analyze_text(text: str) -> dict[str, Any]:
    """Return word statistics and the longest sentence for ``text``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    words = _tokenize(text)
    total = len(words)
    counts = Counter(words)

    # Counter.most_common keeps first-seen order for equal counts,
    # so the result is deterministic.
    top_five = [{"word": w, "count": c} for w, c in counts.most_common(5)]

    average_length = round(sum(len(w) for w in words) / total, 2) if total else 0.0

    longest: dict[str, Any] | None = None
    for raw in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = raw.strip()
        if not sentence:
            continue
        word_count = len(_WORD_RE.findall(sentence))
        if word_count == 0:
            continue  # e.g. "..." or "?!"
        if longest is None or word_count > longest["word_count"]:
            longest = {"text": sentence, "word_count": word_count}

    return {
        "total_words": total,
        "unique_words": len(counts),
        "top_five_words": top_five,
        "average_word_length": average_length,
        "longest_sentence": longest,
    }


# ---------------------------------------------------------------------------
# Transaction analysis
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("transaction_id", "customer_id", "amount", "timestamp")


def _valid_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, int)


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def _validate_record(record: Any) -> list[str]:
    """Return a list of problems with ``record`` (empty list means valid)."""
    if not isinstance(record, dict):
        return ["record is not an object"]

    errors: list[str] = []
    for field in _REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            errors.append(f"missing field: {field}")
    if errors:
        return errors

    if not _valid_id(record["transaction_id"]):
        errors.append("transaction_id must be a non-empty string or integer")
    if not _valid_id(record["customer_id"]):
        errors.append("customer_id must be a non-empty string or integer")

    amount = record["amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        errors.append("amount must be a number")
    elif not math.isfinite(amount):
        errors.append("amount must be finite")
    elif amount <= 0:
        errors.append("amount must be greater than zero")

    if not _valid_timestamp(record["timestamp"]):
        errors.append("timestamp must be a valid ISO 8601 string")

    return errors


def analyze_transactions(records: list[Any]) -> dict[str, Any]:
    """Return duplicates, revenue, top customers and invalid records."""
    if not isinstance(records, list):
        raise TypeError("records must be a list")

    invalid: list[dict[str, Any]] = []
    first_seen: dict[str, int] = {}  # transaction_id -> index of first copy
    duplicate_indexes: dict[str, list[int]] = {}
    spend: dict[str, Decimal] = {}
    tx_count: Counter[str] = Counter()
    revenue = Decimal("0")
    valid_count = 0

    for index, record in enumerate(records):
        errors = _validate_record(record)
        if errors:
            invalid.append({"index": index, "record": record, "errors": errors})
            continue

        valid_count += 1
        tx_id = str(record["transaction_id"])

        if tx_id in first_seen:
            duplicate_indexes.setdefault(tx_id, [first_seen[tx_id]]).append(index)
            continue
        first_seen[tx_id] = index

        amount = Decimal(str(record["amount"]))
        customer = str(record["customer_id"])
        revenue += amount
        spend[customer] = spend.get(customer, Decimal("0")) + amount
        tx_count[customer] += 1

    # Highest spend first; customer id as a stable tie-breaker.
    ranked = sorted(spend.items(), key=lambda item: (-item[1], item[0]))[:3]
    top_customers = [
        {
            "customer_id": customer,
            "total_spent": float(round(total, 2)),
            "transaction_count": tx_count[customer],
        }
        for customer, total in ranked
    ]

    duplicates = [
        {"transaction_id": tx_id, "occurrences": len(idx), "indexes": idx}
        for tx_id, idx in duplicate_indexes.items()
    ]

    return {
        "total_records": len(records),
        "valid_records": valid_count,
        "invalid_count": len(invalid),
        "duplicate_count": sum(len(d["indexes"]) - 1 for d in duplicates),
        "total_revenue": float(round(revenue, 2)),
        "top_customers": top_customers,
        "duplicates": duplicates,
        "invalid_records": invalid,
    }