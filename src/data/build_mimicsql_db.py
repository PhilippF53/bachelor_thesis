"""Rebuild the MIMICSQL database (100 sampled admissions) following TREQS.

TREQS (data/mimicsql/TREQS/evaluation/process_mimic_db.py) builds the database from
the MIMIC-III CSV files. We only have MIMIC-III as SQLite (`data/mimic.db`), so the
required tables are exported to CSV in the original MIMIC-III layout first, then the
unmodified TREQS table builders are imported and run. The sampling step of
process_mimic_db.py runs on import, so it is replicated in `sample_admissions`.

Patient data is only processed in code; nothing but row counts is logged.
"""

import contextlib
import csv
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas
from loguru import logger

from src.config import Config

SOURCE_TABLES = [
    "PATIENTS",
    "ADMISSIONS",
    "DIAGNOSES_ICD",
    "D_ICD_DIAGNOSES",
    "PROCEDURES_ICD",
    "D_ICD_PROCEDURES",
    "PRESCRIPTIONS",
    "LABEVENTS",
    "D_LABITEMS",
]
MIMICSQL_TABLES = ["DEMOGRAPHIC", "DIAGNOSES", "PROCEDURES", "PRESCRIPTIONS", "LAB"]
SAMPLE_SIZE = 100
SAMPLE_SEED = 0


def _format_value(value, declared_type: str) -> str:
    if value is None:
        return ""
    # MIMIC-III CSVs store integer ids without decimals; the SQLite import turned
    # nullable integer columns into REAL (e.g. LABEVENTS.HADM_ID = 123.0).
    if isinstance(value, float) and declared_type == "REAL" and value.is_integer():
        return str(int(value))
    return str(value)


def export_table_csv(source: sqlite3.Connection, table: str, out_file: Path) -> int:
    columns = source.execute(f'PRAGMA table_info("{table}")').fetchall()
    names = [col[1] for col in columns]
    types = [col[2].upper() for col in columns]
    count = 0
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(names)
        # rowid order equals the order of the original CSV import
        for row in source.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
            writer.writerow([_format_value(v, t) for v, t in zip(row, types)])
            count += 1
    return count


@contextlib.contextmanager
def _treqs_context(treqs_dir: Path):
    """TREQS imports `process_mimic_db.*` and reads id2name.csv relative to the cwd."""
    old_cwd = os.getcwd()
    sys.path.insert(0, str(treqs_dir))
    os.chdir(treqs_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(treqs_dir))


def build_full_tables(treqs_dir: Path, csv_dir: Path, out_dir: Path) -> None:
    with _treqs_context(treqs_dir):
        from process_mimic_db.process_tables import (
            build_demographic_table,
            build_diagnoses_table,
            build_lab_table,
            build_prescriptions_table,
            build_procedures_table,
        )

        conn = sqlite3.connect(out_dir / "mimic_all.db")
        build_demographic_table(str(csv_dir), str(out_dir), conn)
        build_diagnoses_table(str(csv_dir), str(out_dir), conn)
        build_procedures_table(str(csv_dir), str(out_dir), conn)
        build_prescriptions_table(str(csv_dir), str(out_dir), conn)
        build_lab_table(str(csv_dir), str(out_dir), conn)
        conn.close()


def sample_admissions(out_dir: Path, target_db: Path) -> None:
    """Same steps as the sampling part of TREQS process_mimic_db.py."""
    conn = sqlite3.connect(target_db)
    data_demo = pandas.read_csv(out_dir / "DEMOGRAPHIC.csv")
    data_demo_sample = data_demo.sample(SAMPLE_SIZE, random_state=SAMPLE_SEED)
    data_demo_sample.to_sql("DEMOGRAPHIC", conn, if_exists="replace", index=False)
    sampled_id = data_demo_sample["HADM_ID"].values

    for table in MIMICSQL_TABLES[1:]:
        logger.info(f"Sampling {table}")
        data_input = pandas.read_csv(out_dir / f"{table}.csv")
        data_filter = [data_input.query("HADM_ID==" + str(itm)) for itm in sampled_id]
        data_out = pandas.concat(data_filter, ignore_index=True)
        data_out.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()


def add_indexes(target_db: Path) -> None:
    conn = sqlite3.connect(target_db)
    for table in MIMICSQL_TABLES:
        for col in ("HADM_ID", "SUBJECT_ID"):
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_{col}" ON "{table}" ("{col}")')
    conn.commit()
    conn.close()


def build_mimicsql_db(config: Config, force: bool = False, keep_intermediate: bool = False) -> Path:
    paths = config.paths
    target_db = config.db_path
    if target_db.exists() and not force:
        logger.info(f"{target_db} exists, skipping (use --force to rebuild)")
        return target_db

    build_dir = paths.derived_dir / "_build"
    csv_dir, out_dir = build_dir / "mimic_csv", build_dir / "treqs_out"
    for d in (csv_dir, out_dir, target_db.parent):
        d.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(f"file:{paths.source_db.resolve()}?mode=ro", uri=True)
    source_counts = {}
    for table in SOURCE_TABLES:
        logger.info(f"Exporting {table}")
        source_counts[table] = export_table_csv(source, table, csv_dir / f"{table}.csv")
    source.close()

    treqs_dir = paths.treqs_dir.resolve()
    build_full_tables(treqs_dir, csv_dir.resolve(), out_dir.resolve())

    tmp_db = target_db.with_suffix(".sqlite.tmp")
    tmp_db.unlink(missing_ok=True)
    sample_admissions(out_dir, tmp_db)
    add_indexes(tmp_db)
    tmp_db.replace(target_db)

    conn = sqlite3.connect(f"file:{target_db.resolve()}?mode=ro", uri=True)
    counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in MIMICSQL_TABLES}
    conn.close()
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_db": str(paths.source_db),
        "source_row_counts": source_counts,
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "row_counts": counts,
    }
    (target_db.parent / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.success(f"Built {target_db}: {counts}")

    if not keep_intermediate:
        shutil.rmtree(build_dir)
    return target_db
