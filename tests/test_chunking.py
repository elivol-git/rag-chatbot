from src.chunking import split_text


def test_empty_text_gives_no_chunks():
    assert split_text("", 100, 10) == []
    assert split_text("   \n\n  ", 100, 10) == []


def test_short_text_is_one_chunk():
    assert split_text("A single short paragraph.", 100, 10) == [
        "A single short paragraph."
    ]


def test_paragraphs_are_packed_up_to_chunk_size():
    text = "\n\n".join(["word " * 20] * 6)  # ~100 chars per paragraph
    chunks = split_text(text, 250, 0)
    assert len(chunks) > 1
    assert all(len(chunk) <= 250 for chunk in chunks)


def test_overlap_carries_previous_tail():
    text = "\n\n".join([f"Paragraph {i} " + "x" * 200 for i in range(4)])
    overlap = 40
    chunks = split_text(text, 300, overlap)
    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        # The tail of the previous chunk reappears at the head of the next one.
        assert previous[-overlap:].lstrip().split()[0] in current[:overlap + 20]


def test_oversized_paragraph_is_split_on_sentences():
    paragraph = " ".join(f"Sentence number {i} about vaults." for i in range(60))
    chunks = split_text(paragraph, 200, 0)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_no_content_is_lost():
    text = "\n\n".join(f"Fact {i}: domes span space." for i in range(30))
    joined = " ".join(split_text(text, 200, 20))
    for i in range(30):
        assert f"Fact {i}:" in joined


def test_invalid_parameters_raise():
    for bad in ((0, 0), (100, 100), (100, -1)):
        try:
            split_text("text", *bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
