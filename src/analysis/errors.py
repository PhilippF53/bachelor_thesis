"""Rule-based error categories for wrong predictions (exec_acc == 0).

Each error gets one main category (first matching rule, in CATEGORIES order) and a list
of all semantic differences found (flags).
"""

from dataclasses import dataclass, field

from sqlglot import exp, parse_one

CATEGORIES = [
    "no_sql",
    "timeout",
    "syntax_error",
    "dialect_error",
    "unknown_table",
    "unknown_column",
    "other_exec_error",
    "wrong_table",
    "join_error",
    "wrong_column",
    "aggregation_error",
    "distinct_error",
    "value_case",
    "value_error",
    "operator_error",
    "other_wrong_result",
]

_SYNTAX_MARKERS = ("near ", "incomplete input", "unrecognized token", "syntax error", "Warning:")
_CMP_OPS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Between)


@dataclass
class SqlFeatures:
    tables: set = field(default_factory=set)
    columns: set = field(default_factory=set)
    aggs: list = field(default_factory=list)
    distinct_aggs: int = 0
    joins: int = 0
    join_keys: set = field(default_factory=set)
    literals: list = field(default_factory=list)
    operators: list = field(default_factory=list)


def features(sql: str, schema_columns: set[str]) -> SqlFeatures | None:
    try:
        ast = parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    f = SqlFeatures()
    f.tables = {t.name.upper() for t in ast.find_all(exp.Table)}
    for col in ast.find_all(exp.Column):
        name = col.name.upper()
        # "value" in double quotes is a string literal in SQLite if no such column exists
        if col.this.args.get("quoted") and not col.table and name not in schema_columns:
            f.literals.append(col.name)
        else:
            f.columns.add(name)
    for agg in ast.find_all(exp.AggFunc):
        f.aggs.append(type(agg).__name__.upper())
        if agg.find(exp.Distinct):
            f.distinct_aggs += 1
    f.aggs.sort()
    joins = list(ast.find_all(exp.Join))
    f.joins = len(joins)
    for j in joins:
        on = j.args.get("on")
        if on is not None:
            f.join_keys |= {c.name.upper() for c in on.find_all(exp.Column)}
    f.literals += [str(lit.this) for lit in ast.find_all(exp.Literal)]
    f.literals.sort()
    f.operators = sorted(type(op).__name__ for op in ast.find_all(*_CMP_OPS))
    return f


def _exec_error_category(error: str) -> str:
    if error == "no sql":
        return "no_sql"
    if error == "timeout":
        return "timeout"
    if any(m in error for m in _SYNTAX_MARKERS):
        return "syntax_error"
    if "no such function" in error:
        return "dialect_error"
    if "no such table" in error:
        return "unknown_table"
    if "no such column" in error:
        return "unknown_column"
    return "other_exec_error"


def categorize(pred_sql: str | None, pred_error: str | None, gold_sql: str,
               schema_columns: set[str]) -> tuple[str, list[str]]:
    if not pred_sql:
        return "no_sql", ["no_sql"]
    if pred_error:
        cat = _exec_error_category(pred_error)
        return cat, [cat]

    p, g = features(pred_sql, schema_columns), features(gold_sql, schema_columns)
    if p is None or g is None:
        return "other_wrong_result", ["unparsed"]
    flags = []
    if p.tables != g.tables:
        flags.append("wrong_table")
    if p.joins != g.joins or (p.join_keys and g.join_keys and p.join_keys != g.join_keys):
        flags.append("join_error")
    if p.columns != g.columns:
        flags.append("wrong_column")
    if p.aggs != g.aggs:
        flags.append("aggregation_error")
    elif p.distinct_aggs != g.distinct_aggs:
        flags.append("distinct_error")
    if p.literals != g.literals:
        same_ignoring_case = sorted(v.lower() for v in p.literals) == sorted(v.lower() for v in g.literals)
        flags.append("value_case" if same_ignoring_case else "value_error")
    if p.operators != g.operators:
        flags.append("operator_error")
    if not flags:
        flags = ["other_wrong_result"]
    main = next(c for c in CATEGORIES if c in flags)
    return main, flags
