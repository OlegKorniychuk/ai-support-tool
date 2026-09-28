"""KB article parser/loader tests. tmp_path fixtures, no real data/kb directory."""

import pytest

from support_ai.kb.loader import load_articles, parse_article

GOOD_ARTICLE = """id: edit-birth-data
title: Edit birth date, time and place
tags: profile, birth data
---
Open Profile -> Birth details and tap the field you want to change.
"""


def test_parse_article_good_article():
    article = parse_article(GOOD_ARTICLE, source="edit-birth-data.md")

    assert article.id == "edit-birth-data"
    assert article.title == "Edit birth date, time and place"
    assert article.tags == ["profile", "birth data"]
    assert article.text == "Open Profile -> Birth details and tap the field you want to change."


def test_parse_article_tags_stripped_and_empty_entries_dropped():
    text = "id: a\ntitle: A\ntags: one, , two,\n---\nBody text.\n"
    article = parse_article(text, source="a.md")
    assert article.tags == ["one", "two"]


def test_parse_article_tags_key_absent_means_empty_list():
    text = "id: a\ntitle: A\n---\nBody text.\n"
    article = parse_article(text, source="a.md")
    assert article.tags == []


@pytest.mark.parametrize(
    "text,match",
    [
        ("id: a\ntitle: A\nBody with no separator.\n", "separator"),
        ("id: a\ntitle: A\nbogus_key: x\n---\nBody.\n", "unknown header key"),
        ("title: A\n---\nBody.\n", "missing required"),
        ("id: a\n---\nBody.\n", "missing required"),
        ("id: a\ntitle: A\n---\n   \n", "empty body"),
        ("id: a\ntitle: A\nno colon here\n---\nBody.\n", "invalid header line"),
    ],
)
def test_parse_article_error_cases(text, match):
    with pytest.raises(ValueError, match=match):
        parse_article(text, source="broken.md")


def test_parse_article_error_names_the_source():
    with pytest.raises(ValueError, match="broken.md"):
        parse_article("no separator here", source="broken.md")


def test_load_articles_reads_all_md_files_and_returns_sorted_by_id(tmp_path):
    (tmp_path / "b-article.md").write_text("id: b-article\ntitle: B\n---\nBody B.\n")
    (tmp_path / "a-article.md").write_text("id: a-article\ntitle: A\n---\nBody A.\n")

    articles = load_articles(tmp_path)

    assert [a.id for a in articles] == ["a-article", "b-article"]


def test_load_articles_id_must_equal_filename_stem(tmp_path):
    (tmp_path / "wrong-name.md").write_text("id: right-name\ntitle: T\n---\nBody.\n")

    with pytest.raises(ValueError, match="wrong-name"):
        load_articles(tmp_path)
