from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ts_render import RenderConfig, parse_ot, render_fragment, synthetic_fragments


def test_parse_ot_string_and_list():
    assert parse_ot("[1.0, 2.5, 3]") == [1.0, 2.5, 3.0]
    assert parse_ot([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert parse_ot(np.array([1.0, 2.0])) == [1.0, 2.0]


def test_parse_ot_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_ot("")
    with pytest.raises(TypeError):
        parse_ot(42)


def test_render_writes_png(tmp_path: Path):
    series = list(np.sin(np.linspace(0, 6.28, 48)) + 10)
    out = tmp_path / "f.png"
    res = render_fragment(series, out_path=out, config=RenderConfig(dpi=80))
    assert res == out
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError):
        render_fragment([])
    with pytest.raises(ValueError):
        render_fragment([1.0, float("nan"), 2.0])


def test_synthetic_fragments_have_varied_lengths():
    frags = list(synthetic_fragments(n=6, seed=1))
    assert len(frags) == 6
    assert {f.length for f in frags} == {24, 48, 96}
    for f in frags:
        assert len(f.series) == f.length
