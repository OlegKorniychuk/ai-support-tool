"""Smoke tests for the MVP 1 Streamlit pages via `streamlit.testing.v1.AppTest`.

These only assert that each page script runs without raising. They do not simulate
widget interaction (e.g. clicking Classify), since that would call the real classify()
pipeline; that path is covered by tests/mvp1/test_classify.py with FakeProvider instead.
The shared home page's smoke test lives in tests/core/test_home_page.py instead, since
`streamlit_app.py` isn't classifier-owned.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_classifier_page_runs_without_exceptions():
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "1_Classifier.py"))
    at.run()
    assert not at.exception


def test_eval_page_runs_without_exceptions():
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "2_Eval.py"))
    at.run()
    assert not at.exception
