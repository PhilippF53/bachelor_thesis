"""EX and EM for single samples, reusing the NL2SQL360 Spider components.

Differences to `SpiderEXEMEvaluator`: DISTINCT is kept, the timeout is enforced
(DBService), failing gold queries are reported instead of asserting, and Spider parse
errors are flagged instead of silently becoming an empty SQL.
"""

from dataclasses import dataclass

from nl2sql360.evaluator.test_suite_sql_eval.evaluation import (
    Evaluator,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    rebuild_sql_col,
    rebuild_sql_val,
)
from nl2sql360.evaluator.test_suite_sql_eval.exec_eval import postprocess, result_eq

from src.eval.hardness import load_spider_schema, parse_spider_sql
from src.services.database_service import DBService, ExecResult


@dataclass
class SampleScore:
    exec_acc: int | None  # None: gold not executable, sample excluded
    exact_acc: int | None  # None: gold not parseable by the Spider parser
    pred_ok: bool
    pred_error: str | None
    pred_row_count: int | None
    pred_runtime_s: float
    gold_row_count: int | None
    em_parse_error_gold: str | None
    em_parse_error_pred: str | None


class SampleEvaluator:
    def __init__(self, db_path, tables_json_path, db_id: str, timeout_s: float) -> None:
        self.db = DBService(db_path, timeout_s=timeout_s)
        self.schema = load_spider_schema(db_path)
        self.kmap = build_foreign_key_map_from_json(str(tables_json_path))[db_id]
        self.evaluator = Evaluator()
        self._gold_cache: dict[str, ExecResult] = {}  # in memory only

    def gold_result(self, gold: str) -> ExecResult:
        if gold not in self._gold_cache:
            self._gold_cache[gold] = self.db.execute_sql(postprocess(gold))
        return self._gold_cache[gold]

    def exact_match(self, pred: str, gold: str) -> tuple[int | None, str | None, str | None]:
        g_sql, g_err = parse_spider_sql(self.schema, gold)
        if g_err:
            return None, g_err, None
        if not pred:
            return 0, None, None  # no prediction: wrong, but not a parse error of the prediction
        p_sql, p_err = parse_spider_sql(self.schema, pred)
        g_sql = rebuild_sql_col(build_valid_col_units(g_sql["from"]["table_units"], self.schema),
                                rebuild_sql_val(g_sql), self.kmap)
        p_sql = rebuild_sql_col(build_valid_col_units(p_sql["from"]["table_units"], self.schema),
                                rebuild_sql_val(p_sql), self.kmap)
        return int(self.evaluator.eval_exact_match(p_sql, g_sql)), None, p_err

    def score(self, pred: str | None, gold: str) -> SampleScore:
        gold_res = self.gold_result(gold)
        pred = pred or ""
        pred_res = self.db.execute_sql(postprocess(pred)) if pred else ExecResult(False, None, "no sql", 0.0)
        if not gold_res.ok:
            exec_acc = None
        elif not pred_res.ok:
            exec_acc = 0
        else:
            exec_acc = int(result_eq(gold_res.rows, pred_res.rows, order_matters="order by" in gold.lower()))
        exact_acc, g_err, p_err = self.exact_match(pred, gold)
        return SampleScore(
            exec_acc=exec_acc,
            exact_acc=exact_acc,
            pred_ok=pred_res.ok,
            pred_error=pred_res.error,
            pred_row_count=pred_res.row_count,
            pred_runtime_s=round(pred_res.runtime_s, 4),
            gold_row_count=gold_res.row_count,
            em_parse_error_gold=g_err,
            em_parse_error_pred=p_err,
        )

    def close(self) -> None:
        self.db.close()
