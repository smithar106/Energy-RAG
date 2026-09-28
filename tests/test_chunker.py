from app.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("A short document.")
    assert len(chunks) == 1
    assert chunks[0].index == 0


def test_long_text_splits_and_overlaps():
    text = ". ".join([f"This is sentence number {i} with some words." for i in range(200)])
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    # chunks are indexed sequentially
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_text():
    assert chunk_text("") == []
