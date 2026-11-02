"""Execute and parse all gold queries; document failures.

Only status, row count, error message and runtime are stored, never result rows.
"""

import json
from collections import Counter

import pandas as pd
from loguru import logger
from nl2sql360.evaluator.test_suite_sql_eval.exec_eval import result_eq
from nl2sql360.parser import SQLParser
from tqdm import tqdm

from src.config import Config
from src.eval.hardness import HARDNESS_ORDER, load_spider_schema, parse_spider_sql, spider_hardness
from src.services.database_service import DBService


def gold_check_file(config: Config, split: str):
    return config.paths.split_results_dir(split) / "gold_check.csv"


def check_gold(config: Config) -> pd.DataFrame:
    db = DBService(config.db_path, timeout_s=config.execution.timeout_s)
    schema = load_spider_schema(config.db_path)
    records = []
    for split in config.dataset.splits:
        samples = json.loads(config.paths.split_file(config.dataset.name, split).read_text())
        for s in tqdm(samples, desc=f"gold {split}"):
            res = db.execute_sql(s["sql"])
            norm = db.execute_sql(s["sql_norm"])
            norm_equal = (
                res.ok and norm.ok
                and result_eq(res.rows, norm.rows, order_matters="order by" in s["sql"].lower())
            )
            try:
                SQLParser(s["sql"], dialect="sqlite")
                sqlglot_error = None
            except Exception as e:
                sqlglot_error = f"{type(e).__name__}: {e}"[:300]
            _, spider_error = parse_spider_sql(schema, s["sql"])
            records.append({
                "split": split,
                "id": s["id"],
                "key": s["key"],
                "exec_ok": res.ok,
                "error": res.error,
                "row_count": res.row_count,
                "empty_result": res.ok and res.row_count == 0,
                "runtime_s": round(res.runtime_s, 4),
                "sql_norm_equal": norm_equal,
                "sqlglot_error": sqlglot_error,
                "spider_parse_error": spider_error,
                "spider_hardness": spider_hardness(schema, s["sql"]),
            })
    db.close()

    df = pd.DataFrame(records)
    for split, part in df.groupby("split", sort=False):
        path = gold_check_file(config, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False)
    write_report(config, df)
    return df


def _error_kind(error: str) -> str:
    for prefix in ("no such column", "no such table", "near", "timeout"):
        if error.startswith(prefix):
            return error if prefix == "no such column" else prefix
    return error[:80]


def write_report(config: Config, df: pd.DataFrame) -> None:
    lines = ["# Gold-SQL-Prüfung (MIMICSQL auf nachgebauter TREQS-DB)", ""]
    lines.append(f"Datenbank: `{config.db_path}`, Timeout {config.execution.timeout_s}s.")
    lines.append("")
    summary = df.groupby("split").agg(
        n=("key", "size"),
        exec_ok=("exec_ok", "sum"),
        empty_result=("empty_result", "sum"),
        sql_norm_equal=("sql_norm_equal", "sum"),
        sqlglot_error=("sqlglot_error", lambda x: x.notna().sum()),
        spider_parse_error=("spider_parse_error", lambda x: x.notna().sum()),
        median_runtime_s=("runtime_s", "median"),
    ).reindex(config.dataset.splits)
    lines += [summary.to_markdown(), ""]
    lines.append(
        "`empty_result` = Gold liefert 0 Zeilen (Plausibilitätstest für den DB-Nachbau). "
        "`sql_norm_equal` = Variante mit einfachen Anführungszeichen (für Few-Shot-Beispiele) "
        "liefert dasselbe Ergebnis."
    )
    lines.append("")

    failed = df[~df["exec_ok"]]
    lines.append(f"## Ausführungsfehler ({len(failed)})")
    lines.append("")
    if len(failed):
        kinds = Counter(_error_kind(e) for e in failed["error"])
        lines.append(pd.DataFrame(kinds.most_common(), columns=["Fehler", "Anzahl"]).to_markdown(index=False))
        lines.append("")
        lines.append("Betroffene Keys: siehe `results/<split>/gold_check.csv` (`exec_ok == False`). "
                     "Diese Queries werden von der Evaluation ausgeschlossen.")
    lines.append("")

    parse_failed = df[df["spider_parse_error"].notna()]
    lines.append(f"## Spider-Parser (Exact Match) kann Gold nicht verarbeiten ({len(parse_failed)})")
    lines.append("")
    if len(parse_failed):
        kinds = Counter(e[:80] for e in parse_failed["spider_parse_error"])
        lines.append(pd.DataFrame(kinds.most_common(10), columns=["Fehler", "Anzahl"]).to_markdown(index=False))
    lines.append("")

    lines.append("## Verteilung Schwierigkeit")
    lines.append("")
    dist = pd.crosstab(df["spider_hardness"].fillna("parse_error"), df["split"])
    dist = dist.reindex(index=[h for h in HARDNESS_ORDER if h in dist.index],
                        columns=[s for s in config.dataset.splits if s in dist.columns])
    lines.append(dist.to_markdown())
    lines.append("")

    config.paths.docs_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.docs_dir / "gold_check.md").write_text("\n".join(lines))
    logger.success(f"Gold check: {int(df['exec_ok'].sum())}/{len(df)} executable")
