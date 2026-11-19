"""Difficulty labels (Spider hardness) and Spider parsing for EM."""

import re
from copy import deepcopy

from nl2sql360.evaluator.test_suite_sql_eval.evaluation import Evaluator, _EMPTY_SQL
from nl2sql360.evaluator.test_suite_sql_eval.process_sql import Schema, get_schema, get_sql
from sqlglot import exp, parse_one

_HARDNESS_EVALUATOR = Evaluator()
# display order of the labels; parse_error marks gold SQL the Spider parser cannot read
HARDNESS_ORDER = ["easy", "medium", "hard", "extra", "parse_error"]


def load_spider_schema(db_path) -> Schema:
    return Schema(get_schema(str(db_path)))


def spider_canonical(schema: Schema, sql: str) -> str:
    """Rewrite syntax the Spider parser does not support, identically for gold and pred.

    The Spider tokenizer treats every double-quoted token as a string value (so
    TABLE."COLUMN" breaks) and only knows plain JOIN. Quoted identifiers that exist in
    the schema are unquoted; everything else (including "value" literals) is kept.
    """
    names = set(schema.schema) | {c for cols in schema.schema.values() for c in cols}
    try:
        ast = parse_one(sql, dialect="sqlite")
    except Exception:
        return sql
    for ident in ast.find_all(exp.Identifier):
        if ident.args.get("quoted") and ident.name.lower() in names:
            ident.set("quoted", False)
    # Spider does not parse output column aliases (SELECT COUNT(x) AS n); they do not
    # change the result, so they are dropped
    for select in ast.find_all(exp.Select):
        select.set("expressions", [e.this if isinstance(e, exp.Alias) else e for e in select.expressions])
    out = ast.sql(dialect="sqlite")
    return re.sub(r"\b(INNER|LEFT|LEFT OUTER|CROSS)\s+JOIN\b", "JOIN", out, flags=re.IGNORECASE)


def parse_spider_sql(schema: Schema, sql: str) -> tuple[dict, str | None]:
    """Spider parse; returns (sql_dict, error). On error the empty SQL is returned, as in NL2SQL360."""
    try:
        return get_sql(schema, spider_canonical(schema, sql)), None
    except Exception as e:
        return deepcopy(_EMPTY_SQL), f"{type(e).__name__}: {e}"[:300]


def spider_hardness(schema: Schema, sql: str) -> str | None:
    parsed, error = parse_spider_sql(schema, sql)
    return None if error else _HARDNESS_EVALUATOR.eval_hardness(parsed)

