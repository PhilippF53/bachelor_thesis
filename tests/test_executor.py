import json
import sqlite3

from src.eval.executor import SampleEvaluator
from src.scripts.spider_get_tables import dump_db_json_schema

GOLD = 'SELECT COUNT ( DISTINCT LAB."SUBJECT_ID" ) FROM LAB WHERE LAB."LABEL" = "a"'


def _evaluator(tmp_path) -> SampleEvaluator:
    # synthetic toy database, no MIMIC data
    db = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE LAB (SUBJECT_ID INTEGER, LABEL TEXT)")
    conn.executemany("INSERT INTO LAB VALUES (?, ?)", [(1, "a"), (1, "a"), (2, "b")])
    conn.commit()
    conn.close()
    tables = tmp_path / "tables.json"
    tables.write_text(json.dumps([dump_db_json_schema(str(db), "toy")]))
    return SampleEvaluator(db, tables, "toy", timeout_s=5)


def test_missing_prediction_is_wrong_but_no_parse_error(tmp_path):
    evaluator = _evaluator(tmp_path)
    try:
        missing = evaluator.score(None, GOLD)
        correct = evaluator.score("SELECT COUNT(DISTINCT SUBJECT_ID) FROM LAB WHERE LABEL = 'a'", GOLD)
    finally:
        evaluator.close()
    assert (missing.exec_acc, missing.exact_acc, missing.em_parse_error_pred) == (0, 0, None)
    assert (correct.exec_acc, correct.exact_acc, correct.em_parse_error_pred) == (1, 1, None)
