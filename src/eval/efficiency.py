"""Time ratios for VES and R-VES, measured like BIRD/NL2SQL360 (`bird_ves`, `bird_rves`).

Per correct prediction (EX = 1, reuse of our EX as with `reuse_ex=True`) gold and
prediction run alternately `iterations` times, each on a fresh connection; the time ratio
is the mean of gold_time / pred_time after BIRD's 3-sigma outlier filter
(`clean_abnormal`). VES = sqrt(ratio), R-VES = sqrt(reward(ratio)), see src/eval/metrics.py.

Differences to NL2SQL360: the connection is read-only with a real timeout (progress
handler, as in DBService), the measurement runs sequentially instead of in 8 parallel
processes (less timing noise), and VES and R-VES share one measurement instead of
timing every query twice. Results are cached per run in efficiency.csv and only measured
again when prediction, gold SQL or the number of iterations change. Only runtimes are
stored, never result rows.
"""

import hashlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from nl2sql360.evaluator.bird_eval.bird_ves import clean_abnormal


class QueryTimeout(Exception):
    pass


class EfficiencyMeter:
    def __init__(self, db_path: str | Path, timeout_s: float, iterations: int) -> None:
        self.uri = f"file:{Path(db_path).resolve()}?mode=ro"
        self.timeout_s = timeout_s
        self.iterations = iterations

    def _timed(self, sql: str) -> float:
        """Runtime of one execution as in BIRD's execute_sql: connect untimed, then
        cursor, execute, fetchall and close timed."""
        conn = sqlite3.connect(self.uri, uri=True)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        deadline = time.perf_counter() + self.timeout_s
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        start = time.perf_counter()
        try:
            conn.cursor().execute(sql).fetchall()
        except sqlite3.OperationalError as e:
            conn.close()
            if str(e) == "interrupted":
                raise QueryTimeout(sql) from e
            raise
        conn.close()
        return time.perf_counter() - start

    def time_ratio(self, pred: str, gold: str) -> float:
        """Mean gold/pred time ratio of a correct prediction; 0 on timeout or error
        (as BIRD does)."""
        try:
            ratios = []
            for _ in range(self.iterations):
                pred_time = self._timed(pred)
                gold_time = self._timed(gold)
                ratios.append(gold_time / pred_time)
        except (QueryTimeout, sqlite3.Error):
            return 0.0
        kept = clean_abnormal(ratios) or ratios  # all ratios equal: std 0 would drop every value
        return float(sum(kept) / len(kept))


def measurement_hash(pred: str, gold: str, iterations: int) -> str:
    return hashlib.sha256(f"{iterations}\n{pred}\n{gold}".encode()).hexdigest()[:16]


class EfficiencyCache:
    """efficiency.csv of one run: key, hash of (iterations, pred, gold), time ratio."""

    columns = ("key", "measurement_hash", "time_ratio", "measured_at")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            df = pd.read_csv(path, dtype={"key": str, "measurement_hash": str})
            self.rows = {r["key"]: r for r in df.to_dict(orient="records")}

    def get(self, key: str, h: str) -> float | None:
        row = self.rows.get(key)
        return float(row["time_ratio"]) if row is not None and row["measurement_hash"] == h else None

    def put(self, key: str, h: str, ratio: float) -> None:
        self.rows[key] = {"key": key, "measurement_hash": h, "time_ratio": ratio,
                          "measured_at": datetime.now().isoformat(timespec="seconds")}

    def save(self, keep: set[str]) -> None:
        """Write the entries of the current samples (drops samples no longer correct)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r for k, r in self.rows.items() if k in keep]
        pd.DataFrame(rows, columns=list(self.columns)).to_csv(self.path, index=False)
