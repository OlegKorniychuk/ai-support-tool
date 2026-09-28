"""Parses KB articles out of `data/kb/*.md` files.

No YAML dependency: the header format below is deliberately minimal, so a small hand-rolled
parser has fewer moving parts (and failure modes) than pulling in a YAML library for ~20
short files.
"""

from pathlib import Path

from support_ai.kb.base import Article

_REQUIRED_KEYS = {"id", "title"}
_KNOWN_KEYS = _REQUIRED_KEYS | {"tags"}


def parse_article(text: str, *, source: str) -> Article:
    """Parse one article: a `key: value` header, a line that is exactly `---`, then the body.

    Required header keys: `id`, `title`. Optional: `tags` (comma-separated, each entry
    stripped, empty entries dropped). Raises `ValueError` naming `source` for: an unknown
    header key, a missing `---` separator, a missing required key, or an empty body.
    """
    lines = text.splitlines()
    try:
        separator_idx = lines.index("---")
    except ValueError:
        raise ValueError(f"{source}: missing '---' header separator") from None

    header: dict[str, str] = {}
    for line in lines[:separator_idx]:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{source}: invalid header line {line!r} (expected 'key: value')")
        key, _, value = line.partition(":")
        key = key.strip()
        if key not in _KNOWN_KEYS:
            raise ValueError(f"{source}: unknown header key {key!r}")
        header[key] = value.strip()

    missing = _REQUIRED_KEYS - header.keys()
    if missing:
        raise ValueError(f"{source}: missing required header key(s) {sorted(missing)}")

    body = "\n".join(lines[separator_idx + 1 :]).strip()
    if not body:
        raise ValueError(f"{source}: empty body")

    tags = [tag.strip() for tag in header.get("tags", "").split(",") if tag.strip()]

    return Article(id=header["id"], title=header["title"], text=body, tags=tags)


def load_articles(directory: Path) -> list[Article]:
    """Load every `*.md` file in `directory`, parsed by `parse_article`, sorted by id.

    Each file's `id` must equal its filename stem (which also makes ids unique); a mismatch
    raises `ValueError` naming the file.
    """
    articles = []
    for path in sorted(directory.glob("*.md")):
        article = parse_article(path.read_text(encoding="utf-8"), source=str(path))
        if article.id != path.stem:
            raise ValueError(f"{path}: id {article.id!r} does not match filename {path.stem!r}")
        articles.append(article)
    return sorted(articles, key=lambda article: article.id)
