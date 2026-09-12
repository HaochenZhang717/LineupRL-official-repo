import numpy as np

from ts_eval.render_multi import MAX_CHANNELS, render_fragment_multi


def test_render_multi_writes_png(tmp_path):
    series = [
        list(np.sin(np.linspace(0, 6, 50))),
        list(np.cos(np.linspace(0, 6, 50)) * 2),
        list(np.linspace(-1, 1, 50)),
    ]
    out = tmp_path / "multi.png"
    result_path, truncated = render_fragment_multi(series, out_path=out)
    assert result_path == out
    assert out.exists() and out.stat().st_size > 0
    assert truncated is False


def test_render_multi_truncates_over_max_channels(tmp_path):
    series = [[float(i), float(i + 1), float(i + 2)] for i in range(MAX_CHANNELS + 3)]
    out = tmp_path / "many.png"
    _, truncated = render_fragment_multi(series, out_path=out)
    assert truncated is True
    assert out.exists()


def test_render_multi_rejects_empty_channel():
    try:
        render_fragment_multi([[1.0, 2.0], []])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_render_multi_returns_figure_when_no_out_path():
    fig, truncated = render_fragment_multi([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], out_path=None)
    assert truncated is False
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
