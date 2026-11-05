"""NL2SQL360 Core with an evaluation that reads our run directories.

Results go into the regular NL2SQL360 evaluation tables (so leaderboards, filters and
scenarios keep working) plus a per-sample CSV with latency, tokens and diagnostics.
Metrics: EX, EM, soft F1 (src/eval/executor.py) and VES/R-VES (src/eval/efficiency.py).
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
from nl2sql360.evaluator.test_suite_sql_eval.exec_eval import postprocess
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.config import Config
from src.eval.efficiency import EfficiencyCache, EfficiencyMeter, measurement_hash
from src.eval.executor import SampleEvaluator
from src.eval.metrics import rves_score, ves_score
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
        out = results_dir(self.config, split, model, variant)
        gold_by_key = {s["key"]: s["sql"] for s in samples}
        self.add_efficiency(rows, gold_by_key, out / "efficiency.csv", f"{model}/{variant}")
        df = pd.DataFrame(rows)
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
                                      exact_acc=r["exact_acc"], f1=r["f1"], ves=r["ves"], rves=r["rves"]))
            session.commit()
        logger.success(f"{table_name}: EX={df['exec_acc'].mean():.3f} EM={df['exact_acc'].mean():.3f} "
                       f"F1={df['f1'].mean():.3f} VES={df['ves'].mean():.3f} R-VES={df['rves'].mean():.3f}")
        return df

    def add_efficiency(self, rows: list[dict], gold_by_key: dict[str, str], cache_path: Path,
                       label: str) -> None:
        """Adds time_ratio, ves and rves to every row: None if the gold SQL is not
        executable (as EX) or the measurement is off, 0 for wrong predictions."""
        iterations = self.config.evaluation.ves_iterations
        if iterations <= 0:
            for r in rows:
                r.update(time_ratio=None, ves=None, rves=None)
            return
        meter = EfficiencyMeter(self.config.db_path, self.config.execution.timeout_s, iterations)
        cache = EfficiencyCache(cache_path)
        todo = []
        for r in rows:
            if r["exec_acc"] != 1:
                continue
            pred, gold = postprocess(r["pred_sql"]), postprocess(gold_by_key[r["key"]])
            h = measurement_hash(pred, gold, iterations)
            r["time_ratio"] = cache.get(r["key"], h)
            if r["time_ratio"] is None:
                todo.append((r, pred, gold, h))
        for r, pred, gold, h in tqdm(todo, desc=f"VES {label}", disable=not todo):
            r["time_ratio"] = meter.time_ratio(pred, gold)
            cache.put(r["key"], h, r["time_ratio"])
        cache.save({r["key"] for r in rows if r["exec_acc"] == 1})
        if todo:
            logger.info(f"{label}: timed {len(todo)} correct predictions ({iterations} iterations each)")
        for r in rows:
            if r["exec_acc"] is None:
                r.update(time_ratio=None, ves=None, rves=None)
            elif r["exec_acc"] == 0:
                r.update(time_ratio=0.0, ves=0.0, rves=0.0)
            else:
                r.update(time_ratio=round(r["time_ratio"], 4), ves=round(ves_score(r["time_ratio"]), 4),
                         rves=round(rves_score(r["time_ratio"]), 4))
