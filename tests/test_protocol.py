from pathlib import Path

import numpy as np
import pytest

from drdvi_enrc.utils.data import load_dataset
from drdvi_enrc.utils.metrics import matched_score_rows


ROOT = Path(__file__).resolve().parents[1]


def test_stickfigures_primary_protocol_has_two_views():
    features, labels, names = load_dataset(
        "stickfigures", ROOT / "data", {"max_samples": None}, seed=42
    )
    assert features.shape[0] == labels.shape[0] == 900
    assert labels.shape[1] == 2
    assert names == ["upper_body", "lower_body"]


def test_equal_count_matching_rejects_cross_cardinality_assignment():
    labels = np.column_stack((np.arange(12) % 3, np.arange(12) % 2))
    predictions = labels[:, ::-1]
    rows = matched_score_rows("test", labels, predictions, ["three", "two"], True)
    assert [row["n_clusters"] for row in rows] == [3, 2]

    wrong_counts = np.column_stack((predictions[:, 0], predictions[:, 0]))
    with pytest.raises(ValueError, match="equal cluster counts"):
        matched_score_rows("test", labels, wrong_counts, ["three", "two"], True)
