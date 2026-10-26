from task_processing_system.processors import analyze_text, analyze_transactions


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------


def test_text_basic_counts():
    result = analyze_text("The cat sat. The cat ran away fast!")
    assert result["total_words"] == 8
    assert result["unique_words"] == 6  # the, cat, sat, ran, away, fast


def test_text_unique_words_is_case_insensitive():
    result = analyze_text("Apple apple APPLE banana")
    assert result["total_words"] == 4
    assert result["unique_words"] == 2
    assert result["top_five_words"][0] == {"word": "apple", "count": 3}


def test_text_top_five_limited_and_sorted():
    text = "a a a a a b b b b c c c d d e f g"
    top = analyze_text(text)["top_five_words"]
    assert len(top) == 5
    assert [t["word"] for t in top] == ["a", "b", "c", "d", "e"]
    assert [t["count"] for t in top] == [5, 4, 3, 2, 1]


def test_text_top_words_tie_is_deterministic():
    top = analyze_text("zebra apple zebra apple mango")["top_five_words"]
    # equal counts keep first-seen order
    assert [t["word"] for t in top] == ["zebra", "apple", "mango"]


def test_text_average_word_length():
    # lengths: 3, 3, 3 -> 3.0 ; "hi there" -> (2 + 5) / 2 = 3.5
    assert analyze_text("one two six")["average_word_length"] == 3.0
    assert analyze_text("hi there")["average_word_length"] == 3.5


def test_text_punctuation_is_ignored_in_words():
    result = analyze_text("Hello, world! Hello... world?")
    assert result["total_words"] == 4
    assert result["unique_words"] == 2


def test_text_contractions_and_hyphens_stay_whole():
    result = analyze_text("Don't use well-known tricks")
    assert result["total_words"] == 4
    assert {"word": "don't", "count": 1} in result["top_five_words"]


def test_text_longest_sentence():
    text = "Short one. This is a much longer sentence indeed! Mid size here?"
    longest = analyze_text(text)["longest_sentence"]
    assert longest["text"] == "This is a much longer sentence indeed!"
    assert longest["word_count"] == 7


def test_text_longest_sentence_tie_returns_first():
    longest = analyze_text("One two. Three four.")["longest_sentence"]
    assert longest["text"] == "One two."


def test_text_no_terminator_is_single_sentence():
    longest = analyze_text("no punctuation at all here")["longest_sentence"]
    assert longest["text"] == "no punctuation at all here"
    assert longest["word_count"] == 5


def test_text_decimal_numbers_do_not_split_sentence():
    longest = analyze_text("Pi is about 3.14 today.")["longest_sentence"]
    assert longest["word_count"] == 5


def test_text_empty_and_whitespace():
    for text in ("", "   \n\t "):
        result = analyze_text(text)
        assert result["total_words"] == 0
        assert result["unique_words"] == 0
        assert result["top_five_words"] == []
        assert result["average_word_length"] == 0.0
        assert result["longest_sentence"] is None


def test_text_only_punctuation():
    result = analyze_text("... !!! ???")
    assert result["total_words"] == 0
    assert result["longest_sentence"] is None


def test_text_unicode_words():
    result = analyze_text("Café café naïve")
    assert result["total_words"] == 3
    assert result["unique_words"] == 2


def test_text_rejects_non_string():
    try:
        analyze_text(123)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("expected TypeError")


# ---------------------------------------------------------------------------
# Transaction analysis
# ---------------------------------------------------------------------------


def tx(tx_id, customer, amount, ts="2024-01-15T10:30:00Z"):
    return {
        "transaction_id": tx_id,
        "customer_id": customer,
        "amount": amount,
        "timestamp": ts,
    }


def test_tx_total_revenue_and_top_customers():
    records = [
        tx("t1", "alice", 100),
        tx("t2", "bob", 50.5),
        tx("t3", "alice", 25.25),
        tx("t4", "carol", 10),
        tx("t5", "dave", 5),
    ]
    result = analyze_transactions(records)
    assert result["total_records"] == 5
    assert result["valid_records"] == 5
    assert result["total_revenue"] == 190.75
    assert [c["customer_id"] for c in result["top_customers"]] == [
        "alice",
        "bob",
        "carol",
    ]
    assert result["top_customers"][0] == {
        "customer_id": "alice",
        "total_spent": 125.25,
        "transaction_count": 2,
    }


def test_tx_duplicates_detected_and_excluded_from_revenue():
    records = [tx("t1", "alice", 100), tx("t1", "alice", 100), tx("t2", "bob", 20)]
    result = analyze_transactions(records)
    assert result["duplicate_count"] == 1
    assert result["duplicates"] == [
        {"transaction_id": "t1", "occurrences": 2, "indexes": [0, 1]}
    ]
    assert result["total_revenue"] == 120.0
    assert result["valid_records"] == 3


def test_tx_triple_duplicate():
    records = [tx("t1", "a", 10)] * 3
    result = analyze_transactions(records)
    assert result["duplicate_count"] == 2
    assert result["duplicates"][0]["occurrences"] == 3
    assert result["total_revenue"] == 10.0


def test_tx_invalid_records_reported_with_reasons():
    records = [
        tx("t1", "alice", 10),
        {"transaction_id": "t2", "customer_id": "bob"},  # missing fields
        tx("t3", "carol", -5),  # negative
        tx("t4", "dave", 0),  # zero
        tx("t5", "erin", "12.5"),  # numeric string
        tx("t6", "frank", True),  # bool is not a number
        tx("t7", "gina", 10, ts="not-a-date"),
        tx("", "hank", 10),  # empty id
        "garbage",
        None,
        tx("t8", "ivy", float("inf")),
    ]
    result = analyze_transactions(records)
    assert result["valid_records"] == 1
    assert result["invalid_count"] == 10
    assert [r["index"] for r in result["invalid_records"]] == list(range(1, 11))
    assert result["total_revenue"] == 10.0
    assert "missing field: amount" in result["invalid_records"][0]["errors"]


def test_tx_timestamp_formats():
    good = ["2024-01-15T10:30:00Z", "2024-01-15T10:30:00+05:30", "2024-01-15"]
    for i, ts in enumerate(good):
        assert analyze_transactions([tx(f"t{i}", "a", 1, ts)])["valid_records"] == 1
    bad = ["", "yesterday", "2024-13-45T00:00:00Z", 1700000000, None]
    for i, ts in enumerate(bad):
        assert analyze_transactions([tx(f"b{i}", "a", 1, ts)])["invalid_count"] == 1


def test_tx_invalid_record_does_not_count_as_duplicate():
    records = [tx("t1", "a", 10), tx("t1", "a", -10)]
    result = analyze_transactions(records)
    assert result["duplicate_count"] == 0
    assert result["invalid_count"] == 1


def test_tx_empty_list():
    result = analyze_transactions([])
    assert result["total_records"] == 0
    assert result["total_revenue"] == 0.0
    assert result["top_customers"] == []
    assert result["duplicates"] == []
    assert result["invalid_records"] == []


def test_tx_all_invalid():
    result = analyze_transactions([{}, {"amount": 5}])
    assert result["valid_records"] == 0
    assert result["total_revenue"] == 0.0
    assert result["top_customers"] == []


def test_tx_fewer_than_three_customers():
    result = analyze_transactions([tx("t1", "a", 5), tx("t2", "b", 7)])
    assert [c["customer_id"] for c in result["top_customers"]] == ["b", "a"]


def test_tx_tie_in_spending_is_deterministic():
    records = [tx("t1", "zed", 10), tx("t2", "amy", 10)]
    result = analyze_transactions(records)
    assert [c["customer_id"] for c in result["top_customers"]] == ["amy", "zed"]


def test_tx_decimal_precision():
    records = [tx(f"t{i}", "a", 0.1) for i in range(10)]
    assert analyze_transactions(records)["total_revenue"] == 1.0


def test_tx_integer_ids_supported():
    records = [
        {"transaction_id": 1, "customer_id": 7, "amount": 5, "timestamp": "2024-01-01"}
    ]
    result = analyze_transactions(records)
    assert result["valid_records"] == 1
    assert result["top_customers"][0]["customer_id"] == "7"


def test_tx_rejects_non_list():
    try:
        analyze_transactions({"a": 1})  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("expected TypeError")