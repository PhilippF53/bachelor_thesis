"""NL2SQL360 Core with an evaluation that reads our run directories.

Results go into the regular NL2SQL360 evaluation tables (so leaderboards, filters and
scenarios keep working) plus a per-sample CSV with latency, tokens and diagnostics.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger
from nl2sql360.arguments import CoreArguments
from nl2sql360.core import Core
from nl2sql360.database import Base, get_evaluation_model
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.config import Config
from src.eval.executor import SampleEvaluator
from src.eval.nl2sql360_tables import drop_evaluation
from src.inference.extract import extract_sql
from src.inference.runner import load_records, load_samples


def eval_name(model: str, variant: str) -> str:
    # NL2SQL360 interpolates table names unquoted into SQL
    return re.sub(r"\W", "_", f"{model}__{variant}")


def results_dir(config: Config, split: str, model: str, variant: str) -> Path:
    return config.paths.split_results_dir(split) / "runs" / model / variant


class MimicCore(Core):
    def __init__(self, config: Config) -> None:
        super().__init__(CoreArguments(core_dir=str(config.paths.nl2sql360_dir), core_name="nl2sql360",
                                       sql_dialect="SQLite"))
        self.config = config
        self._evaluator: SampleEvaluator | None = None

    @property
    def evaluator(self) -> SampleEvaluator:
        if self._evaluator is None:
            self._evaluator = SampleEvaluator(self.config.db_path, self.config.paths.tables_file,
                                              self.config.dataset.db_id, self.config.execution.timeout_s)
        return self._evaluator

    def evaluate_run(self, split: str, model: str, variant: str, run_path: Path,
                     overwrite: bool = False) -> pd.DataFrame | None:
        dataset_name = f"{self.config.dataset.name}_{split}"
        name = eval_name(model, variant)
        table_name = f"DATASET_{dataset_name}_EVALUATION_{name}"
        samples = load_samples(self.config, split)
        records = load_records(run_path / "raw.jsonl")
        stale = sum(extract_sql(r["content"]) != (r["pred_sql"], r["extract_method"]) for r in records.values())
        if stale:
            logger.warning(f"{model}/{variant}/{split}: {stale} stored predictions differ from the current "
                           "SQL extraction; run `reextract` first")
        missing = [s["key"] for s in samples if s["key"] not in records]
        if missing:
            # partial runs (e.g. --limit test runs) are scored only on the finished samples
            # and are not written into the NL2SQL360 tables
            logger.warning(f"{model}/{variant}/{split}: {len(missing)} samples without prediction")
            samples = [s for s in samples if s["key"] in records]

        rows = []
        for s in tqdm(samples, desc=f"eval {model}/{variant}"):
            rec = records[s["key"]]
            score = self.evaluator.score(rec["pred_sql"], s["sql"])
            rows.append({
                "id": s["id"],
                "key": s["key"],
                "model": model,
                "variant": variant,
                "split": split,
                "pred_sql": rec["pred_sql"],
                "extract_method": rec["extract_method"],
                **asdict(score),
                "latency_s": rec["wall_s"],
                "total_duration_s": rec["total_duration_s"],
                "prompt_tokens": rec["prompt_tokens"],
                "output_tokens": rec["output_tokens"],
                "thinking_chars": len(rec["thinking"] or ""),
                "output_truncated": rec["output_truncated"],
                "spider_hardness": s["spider_hardness"],
            })
        df = pd.DataFrame(rows)
        out = results_dir(self.config, split, model, variant)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_sample.csv", index=False)

        if missing:
            return df
        if table_name in self.models_dict:
            if not overwrite:
                logger.warning(f"{table_name} exists; use --overwrite to replace it")
                return df
            drop_evaluation(self, dataset_name, name)
        model_cls = get_evaluation_model(dataset_name, name)
        self.models_dict[model_cls.__tablename__] = model_cls
        Base.metadata.create_all(self.engine, checkfirst=True)
        with Session(self.engine) as session:
            for r in rows:
                session.add(model_cls(id=r["id"], pred=r["pred_sql"] or "", exec_acc=r["exec_acc"],
                                      exact_acc=r["exact_acc"]))
            session.commit()
        logger.success(f"{table_name}: EX={df['exec_acc'].mean():.3f} EM={df['exact_acc'].mean():.3f}")
        return df
