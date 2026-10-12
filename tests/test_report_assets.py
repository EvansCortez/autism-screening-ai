import pandas as pd

from src.report_assets import configure, figure_label_by_score, write_table


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
