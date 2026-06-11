import csv

from conftest import jrows

from footbench import publish


def test_write_scores_csv_long_form(tmp_path):
    rows = (
        jrows("mA", "j1", 4, 3, 5) + jrows("mB", "j1", 5, 5, 5) + jrows("mA", "j2", 2, 3, 3)
        # j2 never judged mB -> empty cells
    )
    path = tmp_path / "scores.csv"
    publish.write_scores_csv(rows, ["mA", "mB"], ["j1", "j2"], path)
    read = list(csv.reader(open(path)))
    assert read[0] == ["judge", "candidate", "soundness", "priors", "code"]
    assert len(read) == 1 + 2 * 2  # one row per judge x candidate
    assert read[1] == ["j1", "mA", "4", "3", "5"]
    assert read[3] == ["j2", "mA", "2", "3", "3"]
    assert read[4] == ["j2", "mB", "", "", ""]


def test_composite_model_order():
    model_scores = [
        {"model": "mB", "criterion": "composite", "rank": 1, "mean_score": 4.0},
        {"model": "mA", "criterion": "composite", "rank": 2, "mean_score": 3.0},
        {"model": "mA", "criterion": "soundness", "rank": 1, "mean_score": 5.0},
    ]
    order = publish.composite_model_order(model_scores, ("mA", "mB", "mC"))
    assert order == ["mB", "mA", "mC"]  # mC unranked -> appended in config order
