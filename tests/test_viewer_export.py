"""save_complex_browser writes a self-contained page (no notebook front end involved)."""
from pathlib import Path

import pytest

pytest.importorskip("ipywidgets")

from open_hit_expander import visualization as viz


def _with_fake_payload(fn):
    original = viz.browser_payload
    viz.browser_payload = lambda *a, **k: {"jobs": [{"id": "job_1", "models": []}],
                                           "proteinChain": "A", "ligandChain": "B"}
    try:
        return fn()
    finally:
        viz.browser_payload = original


def test_standalone_page_contains_payload_and_cdn_tag(tmp_path):
    path = _with_fake_payload(lambda: viz.save_complex_browser(None, tmp_path / "v.html"))
    text = Path(path).read_text()
    assert "job_1" in text and "__PAYLOAD__" not in text
    assert viz.THREE_DMOL_TAG in text


def test_local_3dmol_is_inlined_for_offline_use(tmp_path):
    js = tmp_path / "3Dmol-min.js"
    js.write_text("window.$3Dmol={createViewer(){}}; /* </script> inside */")
    path = _with_fake_payload(lambda: viz.save_complex_browser(None, tmp_path / "v.html", three_dmol_js=js))
    text = Path(path).read_text()
    assert viz.THREE_DMOL_TAG not in text and "window.$3Dmol" in text
    assert text.count("</script>") == Path(viz.__file__).with_name("complex_browser.html").read_text().count("</script>")


def test_empty_payload_is_an_error(tmp_path):
    original = viz.browser_payload
    viz.browser_payload = lambda *a, **k: {"jobs": []}
    try:
        with pytest.raises(ValueError):
            viz.save_complex_browser(None, tmp_path / "v.html")
    finally:
        viz.browser_payload = original
