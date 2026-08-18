from src.loaders import iter_documents, load_document


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_header_becomes_metadata_and_is_stripped(tmp_path):
    path = _write(
        tmp_path,
        "wiki-dome.md",
        "title: Dome\nsource_url: https://example.org/dome\nlicense: CC BY-SA 4.0\n---\n\nA dome spans space.",
    )
    document = load_document(path, tmp_path)

    assert document.metadata["title"] == "Dome"
    assert document.metadata["source_url"] == "https://example.org/dome"
    assert document.text == "A dome spans space."
    assert "title:" not in document.text
    assert document.source == "wiki-dome.md"


def test_missing_header_falls_back_to_filename_title(tmp_path):
    path = _write(tmp_path, "flying-buttress.txt", "Thrust is carried to a pier.")
    document = load_document(path, tmp_path)

    assert document.metadata["title"] == "Flying Buttress"
    assert document.text == "Thrust is carried to a pier."


def test_empty_and_unsupported_files_are_skipped(tmp_path):
    assert load_document(_write(tmp_path, "empty.md", "   \n"), tmp_path) is None
    assert load_document(_write(tmp_path, "notes.docx", "text"), tmp_path) is None


def test_hash_changes_with_content(tmp_path):
    first = load_document(_write(tmp_path, "a.md", "original"), tmp_path).sha256
    second = load_document(_write(tmp_path, "a.md", "edited"), tmp_path).sha256
    assert first != second


def test_iter_documents_walks_subdirectories(tmp_path):
    (tmp_path / "nested").mkdir()
    _write(tmp_path, "top.md", "top level text")
    _write(tmp_path, "nested/inner.md", "nested text")

    sources = {d.source for d in iter_documents(tmp_path)}
    assert sources == {"top.md", "nested/inner.md"}
