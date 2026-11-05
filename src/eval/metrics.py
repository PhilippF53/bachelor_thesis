"""Soft F1 and the R-VES reward on top of the BIRD components shipped with NL2SQL360.

Soft F1 follows `calculate_f1_score` from NL2SQL360 (BIRD mini-dev) with one change: BIRD
turns both results into sets and pairs the rows by position, and the iteration order of
a set of strings changes between Python processes (hash randomisation). The rows are
sorted before pairing here, so the score is reproducible. Everything else, including the
per-row cell matching (`calculate_row_match`), is the BIRD implementation.
"""

import math

from nl2sql360.evaluator.bird_eval.evaluation_f1 import calculate_row_match


def _row_key(row: tuple) -> tuple:
    # total order over mixed cell types (None, int, float, str)
    return tuple((type(v).__name__, repr(v)) for v in row)


def soft_f1(pred_rows: list | None, gold_rows: list) -> float:
    """BIRD soft F1 between two query results; rows are compared cell by cell, row order
    and duplicate rows are ignored. A failed prediction (None) scores 0."""
    if pred_rows is None:
        return 0.0
    if not pred_rows and not gold_rows:
        return 1.0
    predicted = sorted(set(pred_rows), key=_row_key)
    ground_truth = sorted(set(gold_rows), key=_row_key)

    match_scores, pred_only_scores, truth_only_scores = [], [], []
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):  # rows only in the ground truth
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        match, pred_only, truth_only = calculate_row_match(predicted[i], gt_row)
        match_scores.append(match)
        pred_only_scores.append(pred_only)
        truth_only_scores.append(truth_only)
    for _ in range(len(predicted) - len(ground_truth)):  # rows only in the prediction
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp, fp, fn = sum(match_scores), sum(pred_only_scores), sum(truth_only_scores)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


# R-VES reward classes of `bird_rves.iterated_execute_sql` (lower bound of the time ratio
# gold/pred -> reward); NL2SQL360 does not expose them as a function
RVES_REWARDS = [(2.0, 1.25), (1.0, 1.0), (0.5, 0.75), (0.25, 0.5)]
RVES_MIN_REWARD = 0.25


def rves_reward(time_ratio: float) -> float:
    """Reward for one sample; 0 for wrong or not measured predictions (time_ratio 0)."""
    if time_ratio == 0:
        return 0.0
    for bound, reward in RVES_REWARDS:
        if time_ratio >= bound:
            return reward
    return RVES_MIN_REWARD


def ves_score(time_ratio: float) -> float:
    """Per-sample VES as stored by NL2SQL360 (`VesEvaluator`): sqrt of the time ratio."""
    return math.sqrt(time_ratio)


def rves_score(time_ratio: float) -> float:
    """Per-sample R-VES as stored by NL2SQL360 (`RVesEvaluator`): sqrt of the reward."""
    return math.sqrt(rves_reward(time_ratio))
