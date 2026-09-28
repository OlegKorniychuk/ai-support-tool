"""Smoke test for the shared home page via `streamlit.testing.v1.AppTest`.

Lives under `tests/core/` (not `tests/mvp1/`) because `streamlit_app.py` is the shared
entry point both MVP 1 and MVP 2 list tools on, not classifier-owned. Split out of
`tests/mvp1/test_streamlit_pages.py`, which keeps the Classifier- and Eval-page smoke
tests.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_home_page_runs_without_exceptions():
    at = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"))
    at.run()
    assert not at.exception
