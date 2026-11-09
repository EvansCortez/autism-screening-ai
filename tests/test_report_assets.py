import pandas as pd

from src.report_assets import configure, figure_label_by_score, figure_residual_signal, write_table


def label_tables():
    adult = pd.DataFrame({"aq_score": range(11), "NO": [5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0], "YES": [0] * 7 + [4, 3, 2, 1]})
    child = pd.DataFrame({"score": range(11), "negative": [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0], "positive": [0] * 7 + [4, 3, 2, 1]})
    return {"label_adult": adult, "label_child": child}


def test_figures_are_byte_for_byte_reproducible(tmp_path):
    configure()
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        figure_label_by_score(label_tables(), tmp_path / name)
    files = sorted(path.name for path in (tmp_path / "first").iterdir())
    assert {name.rsplit(".", 1)[1] for name in files} == {"svg", "pdf", "png"}
    for name in files:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_table_writer_emits_csv_markdown_and_latex_with_rounded_floats(tmp_path):
    frame = pd.DataFrame({"Method": ["rule"], "Accuracy": [0.123456], "Repeat SD (not a CI)": [0.0]})
    write_table(frame, tmp_path, "t", "Caption text")
    assert "0.1235" in (tmp_path / "t.md").read_text()
    assert "Caption text" in (tmp_path / "t.md").read_text()
    assert (tmp_path / "t.csv").read_text().startswith("Method,Accuracy")
    assert "\\caption{Caption text}" in (tmp_path / "t.tex").read_text()


def test_residual_figure_shows_the_full_null_range(monkeypatch, tmp_path):
    data = {
        "residual_decomposition": pd.DataFrame({"feature_set": ["main set"], "roc_auc_mean": [0.74]}),
        "permutation_null": pd.DataFrame({"null_min": [0.42], "null_mean": [0.5], "null_max": [0.58], "null_draws": [200]}),
    }
    captured = []

    def capture(fig, directory, name):
        ax = fig.axes[0]
        assert ax.get_xlim()[0] <= 0.42 and ax.get_xlim()[1] >= 0.58
        # axvspan creates the rectangle; the bar is a separate rectangle at the start.
        span = ax.patches[-1]
        captured.append((span.get_x(), span.get_width()))
        import matplotlib.pyplot as plt
        plt.close(fig)

    monkeypatch.setattr("src.report_assets.save", capture)
    configure()
    figure_residual_signal(data, tmp_path)
    assert captured[0][0] == 0.42
    assert abs(captured[0][1] - 0.16) < 1e-12
