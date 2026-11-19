"""Convert MimicSQL2 (jsonl) to the NL2SQL360 sample format and import dev/test."""

import json
import re
from pathlib import Path

from loguru import logger
from nl2sql360.arguments import CoreArguments, DatasetArguments
from nl2sql360.core import Core

from src.config import Config
from src.eval.hardness import load_spider_schema, spider_hardness
from src.eval.nl2sql360_tables import drop_dataset
from src.scripts.spider_get_tables import dump_db_json_schema

_DQ_TOKEN = re.compile(r'"([^"]*)"')


def normalize_literals(sql: str) -> str:
    """Rewrite "value" literals as standard SQL 'value' literals.

    Quoted tokens are paired from the left; a token directly preceded by "." is an
    identifier (MIMICSQL always writes TABLE."COLUMN"), every other one a literal.
    """
    def repl(m: re.Match) -> str:
        if m.start() > 0 and sql[m.start() - 1] == ".":
            return m.group(0)
        return "'" + m.group(1).replace("'", "''") + "'"

    return _DQ_TOKEN.sub(repl, sql)


def load_mimicsql_split(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_split(config: Config, split: str) -> list[dict]:
    raw = load_mimicsql_split(config.paths.mimicsql_dir / f"new_{split}.json")
    schema = load_spider_schema(config.db_path)
    samples = []
    for idx, item in enumerate(raw):
        samples.append({
            "id": idx,
            "key": item["key"],
            "split": split,
            "question": item[config.dataset.question_key],
            "sql": item["sql"],
            "sql_norm": normalize_literals(item["sql"]),
            "db_id": config.dataset.db_id,
            "spider_hardness": spider_hardness(schema, item["sql"]) or "parse_error",
        })
    return samples


def write_tables_json(config: Config) -> None:
    schema = dump_db_json_schema(str(config.db_path), config.dataset.db_id)
    config.paths.tables_file.write_text(json.dumps([schema], indent=2))


def nl2sql360_core(config: Config) -> Core:
    return Core(CoreArguments(core_dir=str(config.paths.nl2sql360_dir), core_name="nl2sql360", sql_dialect="SQLite"))


def nl2sql360_dataset_name(config: Config, split: str) -> str:
    return f"{config.dataset.name}_{split}"


def prepare(config: Config) -> None:
    config.paths.derived_dir.mkdir(parents=True, exist_ok=True)
    for split in config.dataset.splits:
        samples = convert_split(config, split)
        config.paths.split_file(config.dataset.name, split).write_text(json.dumps(samples, indent=1))
        logger.info(f"{split}: {len(samples)} samples")
    write_tables_json(config)

    core = nl2sql360_core(config)
    for split in config.dataset.eval_splits:
        name = nl2sql360_dataset_name(config, split)
        if f"DATASET_{name}" in core.models_dict:
            evaluations = list(core.query_available_evaluations(name)["Evaluation"])
            if evaluations:
                # re-importing would require dropping these evaluations
                logger.warning(f"NL2SQL360 dataset {name} kept: it has {len(evaluations)} evaluations")
                continue
            drop_dataset(core, name)  # re-import so it matches the freshly converted samples
        core.import_dataset(DatasetArguments(
            dataset_name=name,
            dataset_dir=str(config.paths.derived_dir),
            samples_file=config.paths.split_file(config.dataset.name, split).name,
            database_dir="database",
            tables_file=config.paths.tables_file.name,
            question_key="question",
            sql_key="sql",
            db_id_key="db_id",
            sql_complexity_key="spider_hardness",
        ))
