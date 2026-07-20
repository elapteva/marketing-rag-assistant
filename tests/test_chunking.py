import pytest

from src.chunking import split_text


def test_split_text_returns_empty_list_for_blank_text():
    result = split_text("   ")

    assert result == []


def test_split_text_returns_one_chunk_for_short_text():
    text = "This is a short marketing report."

    result = split_text(text, chunk_size=100, overlap=20)

    assert result == [text]


def test_split_text_creates_multiple_chunks_for_long_text():
    text = "Marketing campaign performance improved significantly. " * 20

    result = split_text(text, chunk_size=150, overlap=30)

    assert len(result) > 1
    assert all(chunk.strip() for chunk in result)
    assert all(len(chunk) <= 150 for chunk in result)


def test_split_text_preserves_content():
    text = (
        "The email campaign increased engagement. "
        "The social media campaign generated more clicks. "
        "The search campaign produced the highest return."
    )

    result = split_text(text, chunk_size=90, overlap=20)

    combined_text = " ".join(result)

    assert "email campaign increased engagement" in combined_text
    assert "social media campaign generated more clicks" in combined_text
    assert "search campaign produced the highest return" in combined_text


def test_split_text_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="Invalid chunk settings"):
        split_text("Marketing report", chunk_size=0, overlap=0)


def test_split_text_rejects_negative_overlap():
    with pytest.raises(ValueError, match="Invalid chunk settings"):
        split_text("Marketing report", chunk_size=100, overlap=-1)


def test_split_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError, match="Invalid chunk settings"):
        split_text("Marketing report", chunk_size=100, overlap=100)