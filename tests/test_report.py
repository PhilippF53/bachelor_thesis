import pandas as pd

from src.analysis.report import _ranks, _sorted
from src.config import load_config


def test_hardness_columns_and_models_in_display_order():
    config = load_config("config/config.yaml")
    small, large = config.models[0].name, config.models[-1].name
    df = pd.DataFrame({
        "model": [large, small, large, small],
        "spider_hardness": ["medium", "hard", "easy", "medium"],
        "exec_acc": [1, 0, 1, 1],
    })
    t = _sorted(df.pivot_table(index="model", columns="spider_hardness", values="exec_acc"), _ranks(config))
    assert list(t.columns) == ["easy", "medium", "hard"]
    assert list(t.index) == [small, large]
