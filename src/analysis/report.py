"""Error analysis, summary tables (CSV/JSON) and figures from results/<split>/runs/**/per_sample.csv."""

import json
import re
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


def add_gold_tables(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Tables used by the gold SQL (FROM/JOIN), "|"-separated in schema order."""
    conn = sqlite3.connect(f"file:{config.db_path.resolve()}?mode=ro", uri=True)
    order = [r[0].upper() for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    conn.close()
    tables = {}
    for split in df["split"].unique():
        for smp in load_samples(config, split):
            used = {t.upper() for t in re.findall(r'\b(?:FROM|JOIN)\s+"?(\w+)', smp["sql"], flags=re.IGNORECASE)}
            tables[(split, smp["key"])] = "|".join(t for t in order if t in used)
    return df.assign(gold_tables=[tables[k] for k in zip(df["split"], df["key"])])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # per_sample.csv written before F1/VES/R-VES existed: columns stay empty
    df = df.reindex(columns=list(df.columns) + [c for c in ("f1", "ves", "rves") if c not in df.columns])
    df = df.assign(exec_acc_no_dup=df["exec_acc"].where(~df["train_duplicate"]))
    g = df.groupby(["split", "model", "variant"], sort=False)
    summary = g.agg(
        n=("key", "size"),
        n_scored=("exec_acc", "count"),
        EX=("exec_acc", "mean"),
        EM=("exact_acc", "mean"),
        F1=("f1", "mean"),
        VES=("ves", "mean"),
        **{"R-VES": ("rves", "mean")},
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
    for col in ("EX", "EM", "F1", "VES", "R-VES", "EX_excl_train_dup"):
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


def _heatmap(t: pd.DataFrame, title: str, cbar_label: str, col_labels: list[str] | None = None,
             vmax: float = 100):
    """Annotated heatmap, sequential single hue; rows and columns as given."""
    fig, ax = plt.subplots(figsize=(1.2 * len(t.columns) + 3, 0.45 * len(t.index) + 1.8))
    im = ax.imshow(t.values.astype(float), cmap=SEQ_BLUES, vmin=0, vmax=vmax, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * vmax else TEXT)
    ax.set_xticks(range(len(t.columns)), col_labels or list(t.columns), fontsize=8)
    ax.set_yticks(range(len(t.index)), t.index, fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    return fig, ax


def _stacked_shares(ax, t: pd.DataFrame, colors: list[str], legend_title: str | None = None,
                    legend_y: float = -0.3) -> None:
    """Stacked bars of row shares (rows sum to 100)."""
    bottom = pd.Series(0.0, index=t.index)
    for col, color in zip(t.columns, colors):
        ax.bar(range(len(t.index)), t[col], bottom=bottom, color=color, label=col,
               edgecolor=SURFACE, linewidth=1)
        bottom += t[col]
    ax.set_xticks(range(len(t.index)), t.index, rotation=30, ha="right")
    ax.legend(frameon=False, labelcolor=TEXT_2, ncols=3, loc="upper left", bbox_to_anchor=(0, legend_y),
              title=legend_title, title_fontsize=8, fontsize=8)


def _dumbbell(ax, t: pd.DataFrame, start: str, end: str, labels: tuple[str, str]) -> None:
    """One row per index entry: grey line from `start` to `end`, two coloured points."""
    y = range(len(t.index))[::-1]  # first row on top
    ax.hlines(list(y), t[start], t[end], color=GRID, linewidth=2.5, zorder=1)
    ax.scatter(t[start], list(y), s=40, color=SERIES[1], label=labels[0], zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    ax.scatter(t[end], list(y), s=40, color=SERIES[0], label=labels[1], zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(list(y), t.index, fontsize=8)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2)


# ordinal palettes: light -> dark blue; R-VES diverging around "as fast as gold"
F1_BUCKETS = ["F1 = 0", "0 < F1 < 0,5", "0,5 ≤ F1 < 1", "F1 = 1"]
F1_COLORS = ["#dbe9f6", "#9cc3e5", "#4a90d0", "#1a5aa0"]
RVES_CLASSES = {1.25: "≥ 2 (1,25)", 1.0: "1 bis 2 (1,0)", 0.75: "0,5 bis 1 (0,75)",
                0.5: "0,25 bis 0,5 (0,5)", 0.25: "< 0,25 (0,25)"}
RVES_COLORS = ["#1a5aa0", "#7fb0de", "#f4b183", "#e57a3c", "#b4471a"]
METRICS = ["EX", "EM", "F1", "VES", "R-VES"]


def _f1_bucket(f1: float) -> str:
    if f1 == 0:
        return F1_BUCKETS[0]
    if f1 < 0.5:
        return F1_BUCKETS[1]
    return F1_BUCKETS[2] if f1 < 1 else F1_BUCKETS[3]


def best_runs(s: pd.DataFrame) -> pd.DataFrame:
    """Summary row of the best variant (EX) per model; only runs with the full number of
    questions compete, so a test run with --limit cannot win with a few easy questions."""
    full = s[s["n"] == s["n"].max()]
    return full.loc[full.groupby("model")["EX"].idxmax()]


def value_hint_pairs(config: Config) -> list[tuple[str, str]]:
    """(variant without hints, same variant with value hints) as configured."""
    pairs = []
    for v in config.variants.values():
        if v.values:
            pairs += [(b.name, v.name) for b in config.variants.values()
                      if not b.values and (b.k, b.strategy) == (v.k, v.strategy)]
    return pairs


def solved_by_models(config: Config, sel: pd.DataFrame) -> pd.DataFrame:
    """Per question: how many (and which) models solve it with their selected variant;
    models in report.contaminated_models are left out."""
    d = sel[~sel["model"].isin(config.report.contaminated_models) & sel["exec_acc"].notna()]
    order = {m.name: i for i, m in enumerate(config.models)}
    g = d.groupby(["key", "id", "spider_hardness"])
    return g.agg(
        n_models=("model", "size"),
        n_models_solved=("exec_acc", "sum"),
        models_solved=("model", lambda m: "|".join(sorted(m[d.loc[m.index, "exec_acc"] == 1],
                                                          key=lambda x: order.get(x, len(order))))),
    ).reset_index().astype({"n_models_solved": int}).sort_values("id")


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
    best = _sorted(best_runs(s).set_index("model"), ranks)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _grouped_bars(ax, best[["EX", "EM"]])
    _style(ax, f"EX und EM je Modell ({split}, {which})", "%")
    save(fig, "ex_em")

    # all metrics at a glance; VES can exceed 100 (prediction faster than gold)
    m = best[METRICS]
    if m[["F1", "VES", "R-VES"]].notna().any().any():
        vmax = max(100.0, float(m.max().max()))
        fig, _ = _heatmap(m, f"Metriken je Modell ({split}, {which})", "Wert (%)", vmax=vmax)
        save(fig, "metrics_overview")

    # latency vs EX: one point per model, direct labels
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(best["latency_median_s"], best["EX"], s=64, color=SERIES[0],
               edgecolors=SURFACE, linewidths=2, zorder=3)
    for mdl, r in best.iterrows():
        ax.annotate(mdl, (r["latency_median_s"], r["EX"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=TEXT_2)
    ax.set_xscale("log")
    ax.set_xlabel("Median-Latenz pro Anfrage (s, log)", color=TEXT_2)
    _style(ax, f"Latenz gegen EX ({split}, {which})", "EX (%)")
    save(fig, "latency_vs_ex")

    sel = df.merge(best.reset_index()[["model", "variant"]], on=["model", "variant"])
    # EX by difficulty: columns from easy to hard
    h = _sorted(sel.pivot_table(index="model", columns="spider_hardness", values="exec_acc",
                                aggfunc="mean") * 100, ranks)
    counts = sel.drop_duplicates("key")["spider_hardness"].value_counts()
    fig, _ = _heatmap(h, f"EX nach Schwierigkeit (Spider-Hardness)\n{split}, {which}", "EX (%)",
                      [f"{c}\n(n={counts.get(c, 0)})" for c in h.columns])
    save(fig, "ex_by_spider_hardness")

    # EX by table in the gold SQL (a question counts for every table it uses)
    tab = sel[sel["exec_acc"].notna()].assign(table=sel["gold_tables"].str.split("|")).explode("table")
    t = _sorted(tab.pivot_table(index="model", columns="table", values="exec_acc", aggfunc="mean") * 100,
                ranks)
    tcounts = tab.drop_duplicates(["key", "table"])["table"].value_counts()
    t = t[sorted(t.columns, key=lambda c: -tcounts.get(c, 0))]
    fig, _ = _heatmap(t, f"EX nach Tabelle in der Gold-SQL\n{split}, {which}", "EX (%)",
                      [f"{c}\n(n={tcounts.get(c, 0)})" for c in t.columns])
    save(fig, "ex_by_table")

    # error groups: stacked bars, share of all scored samples
    err = sel[sel["exec_acc"] == 0]
    if len(err):
        t = pd.crosstab(err["model"], err["error_group"])
        t = t.div(sel.groupby("model")["exec_acc"].count(), axis=0).mul(100)
        t = _sorted(t, ranks)[[g for g in GROUP_ORDER if g in t.columns]]
        fig, ax = plt.subplots(figsize=(10, 5))
        _stacked_shares(ax, t, [SERIES[GROUP_ORDER.index(c)] for c in t.columns])
        _style(ax, f"Fehlerkategorien ({split}, {which}, Anteil aller Fragen)", "% der Fragen")
        save(fig, "error_groups")

    # how close are the wrong answers: soft F1 of the EX = 0 answers
    wrong = err[err["f1"].notna()]
    if len(wrong):
        t = pd.crosstab(wrong["model"], wrong["f1"].map(_f1_bucket), normalize="index").mul(100)
        t = _sorted(t.reindex(columns=F1_BUCKETS, fill_value=0), ranks)
        n = wrong.groupby("model").size()
        t.index = [f"{mdl}\n(n={n[mdl]})" for mdl in t.index]
        fig, ax = plt.subplots(figsize=(10, 5.4))
        _stacked_shares(ax, t, F1_COLORS, "Soft F1 der falschen Antwort", legend_y=-0.42)
        _style(ax, f"Wie nah liegen falsche Antworten? ({split}, {which})", "% der falschen Antworten")
        save(fig, "f1_wrong_answers")

    # R-VES reward classes of the correct answers (time ratio gold / prediction)
    right = sel[(sel["exec_acc"] == 1) & sel["rves"].notna()]
    if len(right):
        reward = (right["rves"] ** 2).round(2).map(RVES_CLASSES)
        t = pd.crosstab(right["model"], reward, normalize="index").mul(100)
        t = _sorted(t.reindex(columns=list(RVES_CLASSES.values()), fill_value=0), ranks)
        fig, ax = plt.subplots(figsize=(10, 5))
        _stacked_shares(ax, t, RVES_COLORS, "Laufzeit Gold / Vorhersage (R-VES-Belohnung)")
        _style(ax, f"Effizienz der richtigen Antworten ({split}, {which})", "% der richtigen Antworten")
        save(fig, "rves_rewards")

    # value hints: EX and value-case errors without and with hints, all models
    for base, hinted in value_hint_pairs(config):
        if not {base, hinted} <= set(variants):
            continue
        pair = df[df["variant"].isin([base, hinted]) & df["exec_acc"].notna()]
        # only questions answered in both variants (e.g. while one run is still incomplete)
        both = pair.groupby(["model", "key"])["variant"].transform("nunique") == 2
        pair = pair[both]
        if pair.empty:
            continue
        g = pair.groupby(["model", "variant"])
        t = pd.DataFrame({
            "EX": g["exec_acc"].mean() * 100,
            "value_case": g["error_group"].apply(lambda x: (x == "Wert: Groß-/Kleinschreibung").mean() * 100),
        }).unstack("variant")
        t = _sorted(t, ranks)
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(t.index) + 2), sharey=True)
        for ax, (col, xlabel) in zip(axes, [("EX", "EX (%)"),
                                            ("value_case", "Fehler Wert-Schreibweise (% der Fragen)")]):
            _dumbbell(ax, t[col], base, hinted, (base, hinted))
            ax.set_xlabel(xlabel, color=TEXT_2)
        axes[0].legend(frameon=False, labelcolor=TEXT_2, ncols=2, loc="upper left",
                       bbox_to_anchor=(0, -0.12 - 0.6 / max(len(t.index), 1)))
        axes[0].set_title(f"Wirkung der Wert-Hinweise ({split}, {pair['key'].nunique()} Fragen in beiden Varianten)",
                          loc="left", color=TEXT, fontsize=12)
        save(fig, f"value_hints_{hinted}")

    # train duplicates: EX with and without the questions that also occur in train
    if best["n_train_dup"].sum() > 0:
        t = best[["variant", "EX", "EX_excl_train_dup"]].copy()
        t.index = [f"{mdl} ({r['variant']})" for mdl, r in t.iterrows()]
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(t.index) + 2))
        _dumbbell(ax, t, "EX", "EX_excl_train_dup", ("alle Fragen", "ohne Train-Duplikate"))
        ax.set_xlabel("EX (%)", color=TEXT_2)
        ax.legend(frameon=False, labelcolor=TEXT_2, ncols=2, loc="upper left",
                  bbox_to_anchor=(0, -0.12 - 0.6 / max(len(t.index), 1)))
        n_dup = int(best["n_train_dup"].iloc[0])
        ax.set_title(f"EX mit und ohne die {n_dup} Fragen, die auch in train stehen\n{split}, {which}",
                     loc="left", color=TEXT, fontsize=11)
        save(fig, "ex_train_duplicates")

    # solved by how many models (without contaminated models), stacked by difficulty
    solved = solved_by_models(config, sel)
    if len(solved) and solved["n_models"].max() > 1:
        n_models = int(solved["n_models"].max())
        t = pd.crosstab(solved["n_models_solved"], solved["spider_hardness"])
        t = t.reindex(index=range(n_models + 1), fill_value=0)
        t = t[[c for c in HARDNESS_ORDER if c in t.columns]]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bottom = pd.Series(0, index=t.index)
        for col, color in zip(t.columns, ["#9cc3e5", "#4a90d0", "#1a5aa0", "#0b2f5a"]):
            ax.bar(t.index, t[col], bottom=bottom, color=color, label=col, edgecolor=SURFACE, linewidth=1)
            bottom += t[col]
        for x, v in bottom.items():
            ax.annotate(str(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=TEXT_2)
        ax.set_xticks(list(t.index))
        ax.set_xlabel("Anzahl Modelle, die die Frage richtig beantworten", color=TEXT_2)
        ax.legend(frameon=False, labelcolor=TEXT_2, ncols=4, loc="upper left")
        left_out = [mdl for mdl in config.report.contaminated_models if mdl in set(sel["model"])]
        note = f", ohne {', '.join(left_out)}" if left_out else ""
        _style(ax, f"Von wie vielen Modellen gelöst? ({split}, {which}{note})", "Fragen")
        save(fig, "solved_by_models")


def report(config: Config) -> None:
    df = add_gold_tables(config, add_train_duplicates(config, add_error_categories(config, load_per_sample(config))))
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

        best = best_runs(s)[["model", "variant"]]
        solved_by_models(config, d.merge(best, on=["model", "variant"])).to_csv(
            out / "solved_by_models.csv", index=False)

        plot_split(config, split, d, s, out / "figures")
        logger.success(f"Report for {split} written to {out}")
