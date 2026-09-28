from app.ingestion.chunker import chunk_segments
from app.ingestion.html import Segment


def test_short_text_single_chunk():
    chunks = chunk_segments([Segment(heading="Intro", text="A short paragraph.")])
    assert len(chunks) == 1
    assert chunks[0].section == "Intro"
    assert chunks[0].index == 0


def test_long_text_splits_with_sequential_indices():
    text = "\n\n".join(
        "Sentence number %d with several words in it." % i for i in range(120)
    )
    chunks = chunk_segments(
        [Segment(heading="Body", text=text)], target_words=50, overlap_words=10
    )
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_sections_are_preserved():
    segs = [
        Segment(heading="Alpha", text="alpha " * 300),
        Segment(heading="Beta", text="beta " * 300),
    ]
    chunks = chunk_segments(segs, target_words=100, overlap_words=20)
    sections = {c.section for c in chunks}
    assert sections == {"Alpha", "Beta"}


def test_empty():
    assert chunk_segments([]) == []
    assert chunk_segments([Segment(heading=None, text="")]) == []
