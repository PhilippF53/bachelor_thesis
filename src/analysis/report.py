"""Error analysis, summary tables (CSV/JSON) and figures from results/<split>/runs/**/per_sample.csv."""

import json
import sqlite3

import matplotlib
import pandas as pd
from loguru import logger

from src.analysis.errors import CATEGORIES, categorize
from src.config import Config
from src.eval.hardness import HARDNESS_ORDER
from src.inference.runner import load_samples

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (dataviz skill), categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_BLUES = "Blues"

ERROR_GROUPS = {
    "no_sql": "Ausführung/Syntax",
    "timeout": "Ausführung/Syntax",
    "syntax_error": "Ausführung/Syntax",
    "dialect_error": "Ausführung/Syntax",
    "other_exec_error": "Ausführung/Syntax",
    "unknown_table": "Schema (Tabelle/Spalte/JOIN)",
    "unknown_column": "Schema (Tabelle/Spalte/JOIN)",
    "wrong_table": "Schema (Tabelle/Spalte/JOIN)",
    "join_error": "Schema (Tabelle/Spalte/JOIN)",
    "wrong_column": "Schema (Tabelle/Spalte/JOIN)",
    "aggregation_error": "Aggregation/DISTINCT",
    "distinct_error": "Aggregation/DISTINCT",
    "value_case": "Wert: Groß-/Kleinschreibung",
    "value_error": "Wert: sonstig",
    "operator_error": "Operator/Bedingung",
    "other_wrong_result": "Sonstiges",
}
# also fixes the colour of each group, so dev and test figures match
GROUP_ORDER = ["Ausführung/Syntax", "Schema (Tabelle/Spalte/JOIN)", "Aggregation/DISTINCT",
               "Wert: Groß-/Kleinschreibung", "Wert: sonstig", "Operator/Bedingung", "Sonstiges"]


def load_per_sample(config: Config) -> pd.DataFrame:
    files = sorted(config.paths.results_dir.glob("*/runs/*/*/per_sample.csv"))
    if not files:
        raise SystemExit("no per_sample.csv found; run `evaluate` first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _ranks(config: Config) -> dict[str, dict[str, int]]:
    """Display order per column: splits, models and variants as in the config, hardness
    from easy to extra, error categories in rule order."""
    orders = {
        "split": config.dataset.splits,
        "model": [m.name for m in config.models],
        "variant": list(config.variants),
        "spider_hardness": HARDNESS_ORDER,
        "error_category": CATEGORIES,
    }
    return {col: {v: i for i, v in enumerate(values)} for col, values in orders.items()}


def _sort_key(ranks: dict[str, dict[str, int]]):
    """Key for sort_index/sort_values (applied per column or index level); unknown values last."""
    def key(values):
        order = ranks.get(values.name)
        return values.map(lambda v: order.get(v, len(order))) if order else values

    return key


def _sorted(t: pd.DataFrame, ranks: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Rows and named columns (pivot/crosstab) in display order."""
    key = _sort_key(ranks)
    t = t.sort_index(key=key)
    return t.sort_index(axis=1, key=key) if t.columns.name in ranks else t


def _schema_columns(config: Config) -> set[str]:
    conn = sqlite3.connect(f"file:{config.db_path.resolve()}?mode=ro", uri=True)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    cols = {r[1].upper() for t in tables for r in conn.execute(f'PRAGMA table_info("{t}")')}
    conn.close()
    return cols


def add_error_categories(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    gold_norm = {}
    for split in df["split"].unique():
        gold_norm.update({(split, s["key"]): s["sql_norm"] for s in load_samples(config, split)})
    cols = _schema_columns(config)
    cats, flags = [], []
    for row in df.itertuples():
        if row.exec_acc != 0:
            cats.append(None)
            flags.append(None)
            continue
        pred = row.pred_sql if isinstance(row.pred_sql, str) else None
        err = row.pred_error if isinstance(row.pred_error, str) else None
        cat, fl = categorize(pred, err, gold_norm[(row.split, row.key)], cols)
        cats.append(cat)
        flags.append("|".join(fl))
    df = df.copy()
    df["error_category"] = cats
    df["error_flags"] = flags
    df["error_group"] = df["error_category"].map(ERROR_GROUPS)
    return df


def add_train_duplicates(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Flag samples whose question and gold SQL also appear in train (overlap of the
    MimicSQL2 splits); few_similar_k3 retrieves the identical example for them."""
    train = {(s["question"], s["sql"]) for s in load_samples(config, "train")}
    dup = {}
    for split in df["split"].unique():
        dup.update({(split, s["key"]): (s["question"], s["sql"]) in train for s in load_samples(config, split)})
    return df.assign(train_duplicate=[dup[k] for k in zip(df["split"], df["key"])])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(exec_acc_no_dup=df["exec_acc"].where(~df["train_duplicate"]))
    g = df.groupby(["split", "model", "variant"], sort=False)
    summary = g.agg(
        n=("key", "size"),
        n_scored=("exec_acc", "count"),
        EX=("exec_acc", "mean"),
        EM=("exact_acc", "mean"),
        n_train_dup=("train_duplicate", "sum"),
        EX_excl_train_dup=("exec_acc_no_dup", "mean"),
        latency_mean_s=("latency_s", "mean"),
        latency_median_s=("latency_s", "median"),
        prompt_tokens_mean=("prompt_tokens", "mean"),
        output_tokens_mean=("output_tokens", "mean"),
        no_sql=("extract_method", lambda x: x.isin(["no_sql", "empty"]).sum()),
        output_truncated=("output_truncated", "sum"),
        em_parse_error_pred=("em_parse_error_pred", lambda x: x.notna().sum()),
        em_parse_error_gold=("em_parse_error_gold", lambda x: x.notna().sum()),
    ).reset_index()
    summary["gold_excluded"] = summary["n"] - summary["n_scored"]
    for col in ("EX", "EM", "EX_excl_train_dup"):
        summary[col] = (summary[col] * 100).round(2)
    for cols, digits in ((["latency_mean_s", "latency_median_s"], 3), (["prompt_tokens_mean", "output_tokens_mean"], 1)):
        summary[cols] = summary[cols].round(digits)
    return summary


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2)


def _grouped_bars(ax, table: pd.DataFrame, value_fmt="{:.1f}") -> None:
    n = len(table.columns)
    width = 0.8 / n
    x = range(len(table.index))
    for i, col in enumerate(table.columns):
        pos = [xi - 0.4 + width * (i + 0.5) for xi in x]
        bars = ax.bar(pos, table[col], width=width * 0.92, color=SERIES[i], label=col)
        if n <= 4:
            ax.bar_label(bars, labels=[value_fmt.format(v) if pd.notna(v) else "" for v in table[col]],
                         fontsize=7, color=TEXT_2, padding=2)
    ax.set_xticks(list(x), table.index, rotation=30, ha="right")
    ax.legend(frameon=False, labelcolor=TEXT_2, ncols=min(n, 4), loc="upper left", bbox_to_anchor=(0, -0.25))


def plot_split(config: Config, split: str, df: pd.DataFrame, s: pd.DataFrame, fig_dir) -> None:
    """Figures for one split; df and s contain only this split."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    ranks = _ranks(config)

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=200)
        fig.savefig(fig_dir / f"{name}.svg")
        plt.close(fig)

    # on dev each model is shown with its best variant, on test with the one test variant
    variants = s["variant"].unique()
    which = variants[0] if len(variants) == 1 else "beste Variante je Modell"

    # variants on dev: EX per model and variant
    if len(variants) > 1:
        t = _sorted(s.pivot(index="model", columns="variant", values="EX"), ranks)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        _grouped_bars(ax, t)
        _style(ax, f"Execution Accuracy je Prompt-Variante ({split})", "EX (%)")
        save(fig, "ex_by_variant")
    # EX and EM per model
    best = _sorted(s.loc[s.groupby("model")["EX"].idxmax()].set_index("model"), ranks)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _grouped_bars(ax, best[["EX", "EM"]])
    _style(ax, f"EX und EM je Modell ({split}, {which})", "%")
    save(fig, "ex_em")

    # latency vs EX: one point per model, direct labels
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(best["latency_median_s"], best["EX"], s=64, color=SERIES[0],
               edgecolors=SURFACE, linewidths=2, zorder=3)
    for m, r in best.iterrows():
        ax.annotate(m, (r["latency_median_s"], r["EX"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=TEXT_2)
    ax.set_xscale("log")
    ax.set_xlabel("Median-Latenz pro Anfrage (s, log)", color=TEXT_2)
    _style(ax, f"Latenz gegen EX ({split}, {which})", "EX (%)")
    save(fig, "latency_vs_ex")

    sel = df.merge(best.reset_index()[["model", "variant"]], on=["model", "variant"])
    # EX by difficulty: heatmap, sequential single hue, columns from easy to hard
    h = sel.pivot_table(index="model", columns="spider_hardness", values="exec_acc", aggfunc="mean") * 100
    h = _sorted(h, ranks)
    fig, ax = plt.subplots(figsize=(1.2 * len(h.columns) + 3, 0.45 * len(h.index) + 1.8))
    im = ax.imshow(h.values, cmap=SEQ_BLUES, vmin=0, vmax=100, aspect="auto")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 60 else TEXT)
    counts = sel.drop_duplicates("key")["spider_hardness"].value_counts()
    ax.set_xticks(range(len(h.columns)), [f"{c}\n(n={counts.get(c, 0)})" for c in h.columns], fontsize=8)
    ax.set_yticks(range(len(h.index)), h.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="EX (%)", shrink=0.8)
    ax.set_title(f"EX nach Schwierigkeit (Spider-Hardness)\n{split}, {which}", loc="left", color=TEXT, fontsize=11)
    save(fig, "ex_by_spider_hardness")

    # error groups: stacked bars, share of all scored samples
    err = sel[sel["exec_acc"] == 0]
    if len(err):
        t = pd.crosstab(err["model"], err["error_group"])
        t = t.div(sel.groupby("model")["exec_acc"].count(), axis=0).mul(100)
        t = _sorted(t, ranks)[[g for g in GROUP_ORDER if g in t.columns]]
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = pd.Series(0.0, index=t.index)
        for col in t.columns:
            ax.bar(t.index, t[col], bottom=bottom, color=SERIES[GROUP_ORDER.index(col)], label=col,
                   edgecolor=SURFACE, linewidth=1)
            bottom += t[col]
        ax.set_xticks(range(len(t.index)), t.index, rotation=30, ha="right")
        ax.legend(frameon=False, labelcolor=TEXT_2, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.3))
        _style(ax, f"Fehlerkategorien ({split}, {which}, Anteil aller Fragen)", "% der Fragen")
        save(fig, "error_groups")


def report(config: Config) -> None:
    df = add_train_duplicates(config, add_error_categories(config, load_per_sample(config)))
    ranks = _ranks(config)
    key = _sort_key(ranks)
    summary = summarize(df).sort_values(["split", "model", "variant"], key=key)

    keys = ["model", "variant"]
    for split in summary["split"].unique():
        out = config.paths.split_results_dir(split)
        out.mkdir(parents=True, exist_ok=True)
        d, s = df[df["split"] == split], summary[summary["split"] == split]
        d.sort_values(keys + ["id"], key=key).to_csv(out / "all_samples.csv", index=False)
        s.to_csv(out / "summary.csv", index=False)
        (out / "summary.json").write_text(json.dumps(s.to_dict(orient="records"), indent=2))

        err = d[d["exec_acc"] == 0]
        _sorted(pd.crosstab([err[k] for k in keys], err["error_category"]), ranks).to_csv(
            out / "errors_by_model.csv")
        acc = d.pivot_table(index=keys, columns="spider_hardness", values="exec_acc", aggfunc="mean") * 100
        _sorted(acc, ranks).round(2).to_csv(out / "ex_by_spider_hardness.csv")
        _sorted(pd.crosstab([err[k] for k in keys] + [err["spider_hardness"]], err["error_category"]),
                ranks).to_csv(out / "errors_by_model_spider_hardness.csv")

        plot_split(config, split, d, s, out / "figures")
        logger.success(f"Report for {split} written to {out}")
