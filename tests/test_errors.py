import pytest

from src.analysis.errors import categorize

COLS = {"SUBJECT_ID", "HADM_ID", "GENDER", "AGE", "SHORT_TITLE", "DRUG", "FORMULARY_DRUG_CD"}
GOLD = ("SELECT COUNT ( DISTINCT DEMOGRAPHIC.\"SUBJECT_ID\" ) FROM DEMOGRAPHIC INNER JOIN DIAGNOSES "
        "on DEMOGRAPHIC.HADM_ID = DIAGNOSES.HADM_ID WHERE DIAGNOSES.\"SHORT_TITLE\" = 'Hypertension NOS'")


@pytest.mark.parametrize("pred, error, expected", [
    (None, "no sql", "no_sql"),
    ("SELECT", "incomplete input", "syntax_error"),
    ("SELECT x FROM LAB", "timeout", "timeout"),
    ("SELECT list_count(x) FROM T", "no such function: list_count", "dialect_error"),
    ("SELECT * FROM PATIENTS", "no such table: PATIENTS", "unknown_table"),
    ("SELECT COUNT(DISTINCT SUBJECT_ID) FROM DIAGNOSES WHERE SHORT_TITLE = 'Hypertension NOS'", None,
     "wrong_table"),
    (GOLD.replace("'Hypertension NOS'", "'hypertension nos'"), None, "value_case"),
    (GOLD.replace("'Hypertension NOS'", "'Hypertension'"), None, "value_error"),
    (GOLD.replace("DISTINCT ", ""), None, "distinct_error"),
    (GOLD.replace("'Hypertension NOS'", '"Hypertension NOS"'), None, "other_wrong_result"),
])
def test_categorize(pred, error, expected):
    assert categorize(pred, error, GOLD, COLS)[0] == expected
