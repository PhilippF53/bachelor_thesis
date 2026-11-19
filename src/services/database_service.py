import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:

    ok: bool
    rows: list | None
    error: str | None
    runtime_s: float

    @property
    def row_count(self) -> int | None:
        return None if self.rows is None else len(self.rows)

    @property
    def timed_out(self) -> bool:
        return self.error == "timeout"


class DBService:
    def __init__(self, database_path: str | Path, timeout_s: float = 30) -> None:
        self.database_path = Path(database_path)
        self.timeout_s = timeout_s
        self._conn = sqlite3.connect(f"file:{self.database_path.resolve()}?mode=ro", uri=True)
        self._conn.text_factory = lambda b: b.decode(errors="ignore")
        self._deadline = 0.0
        # asyncio.wait_for (as in the Spider evaluator) cannot stop a blocking sqlite call;
        # the progress handler aborts the statement from inside sqlite instead.
        self._conn.set_progress_handler(lambda: 1 if time.perf_counter() > self._deadline else 0, 10_000)

    def execute_sql(self, sql_statement: str) -> ExecResult:
        start = time.perf_counter()
        self._deadline = start + self.timeout_s
        try:
            rows = self._conn.execute(sql_statement).fetchall()
            return ExecResult(True, rows, None, time.perf_counter() - start)
        except sqlite3.OperationalError as e:
            runtime = time.perf_counter() - start
            if str(e) == "interrupted":
                return ExecResult(False, None, "timeout", runtime)
            return ExecResult(False, None, str(e), runtime)
        except Exception as e:  # sqlite3.Warning for multiple statements, etc.
            return ExecResult(False, None, f"{type(e).__name__}: {e}", time.perf_counter() - start)

    def close(self) -> None:
        self._conn.close()
