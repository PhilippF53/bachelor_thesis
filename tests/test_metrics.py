import math
import sqlite3

import pytest

from src.analysis.report import value_hint_pairs
from src.config import load_config
from src.eval.efficiency import EfficiencyCache, EfficiencyMeter, measurement_hash
from src.eval.metrics import rves_reward, rves_score, soft_f1, ves_score


def test_soft_f1_basic_cases():
    assert soft_f1([(1,)], [(1,)]) == 1.0
    assert soft_f1([(2,)], [(1,)]) == 0.0
    assert soft_f1([], []) == 1.0  # both empty, as in BIRD
    assert soft_f1(None, [(1,)]) == 0.0  # failed prediction
    assert soft_f1([], [(1,)]) == 0.0
    # order and duplicate rows do not matter
    assert soft_f1([("b",), ("a",), ("a",)], [("a",), ("b",)]) == 1.0


def test_soft_f1_partial_credit():
    # one of two cells right in the only row: precision = recall = 0.5
    assert soft_f1([("a", 1)], [("a", 2)]) == pytest.approx(0.5)
    # one extra row: tp 1, fp 1, fn 0
    assert soft_f1([("a",), ("x",)], [("a",)]) == pytest.approx(2 / 3)


def test_soft_f1_independent_of_input_order():
    gold = [("p1", 3), ("p2", 1), ("p3", None)]
    pred = [("p3", None), ("p2", 1), ("p9", 3)]
    assert soft_f1(pred, gold) == soft_f1(list(reversed(pred)), list(reversed(gold)))


def test_rves_reward_classes_as_bird():
    assert rves_reward(0) == 0  # wrong answer
    assert [rves_reward(r) for r in (3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.25, 0.1)] == \
        [1.25, 1.25, 1, 1, 0.75, 0.75, 0.5, 0.5, 0.25]
    assert ves_score(4) == 2
    assert rves_score(1.0) == 1 and rves_score(5) == pytest.approx(math.sqrt(1.25))


def _toy_db(tmp_path):
    db = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE T (A INTEGER)")
    conn.executemany("INSERT INTO T VALUES (?)", [(i,) for i in range(200)])
    conn.commit()
    conn.close()
    return db


def test_efficiency_meter(tmp_path):
    meter = EfficiencyMeter(_toy_db(tmp_path), timeout_s=5, iterations=20)
    ratio = meter.time_ratio("SELECT COUNT(*) FROM T", "SELECT COUNT(*) FROM T")
    assert 0.2 < ratio < 5  # same query: ratio around 1
    assert meter.time_ratio("SELECT nope FROM T", "SELECT COUNT(*) FROM T") == 0
    # read-only: a writing statement fails instead of changing the database
    assert meter.time_ratio("DELETE FROM T", "SELECT COUNT(*) FROM T") == 0
    conn = sqlite3.connect(tmp_path / "toy.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM T").fetchone() == (200,)
    conn.close()


def test_efficiency_cache_invalidates_on_change(tmp_path):
    path = tmp_path / "efficiency.csv"
    h = measurement_hash("SELECT 1", "SELECT 1", 100)
    cache = EfficiencyCache(path)
    cache.put("k1", h, 1.1)
    cache.put("k2", h, 0.9)
    cache.save(keep={"k1"})
    cache = EfficiencyCache(path)
    assert cache.get("k1", h) == pytest.approx(1.1)
    assert cache.get("k2", h) is None  # no longer correct, dropped
    assert cache.get("k1", measurement_hash("SELECT 2", "SELECT 1", 100)) is None
    assert cache.get("k1", measurement_hash("SELECT 1", "SELECT 1", 50)) is None


def test_value_hint_pairs_from_config():
    assert ("few_similar_k3", "few_similar_k3_values") in value_hint_pairs(load_config("config/config.yaml"))
