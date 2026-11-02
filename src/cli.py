import argparse

from src.config import load_config


def _log_through_tqdm() -> None:
    """Route loguru through tqdm.write so log lines do not break the progress bars."""
    from loguru import logger
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mimic-bench")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-db", help="rebuild the MIMICSQL database (TREQS procedure)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    sub.add_parser("prepare", help="convert MimicSQL2, write tables.json, import into NL2SQL360")
    sub.add_parser("check-gold", help="execute and parse all gold queries")

    p = sub.add_parser("infer", help="run inference (resumes unfinished runs)")
    p.add_argument("--split", default="dev")
    p.add_argument("--models", nargs="*", help="model names from the config (default: all, in order)")
    p.add_argument("--variants", nargs="*", help="variant names (default: dev_variants / test_variant)")
    p.add_argument("--limit", type=int, help="only the first N samples (test runs)")

    p = sub.add_parser("evaluate", help="score runs (EX, EM) and store them in NL2SQL360")
    p.add_argument("--split", default="dev")
    p.add_argument("--models", nargs="*")
    p.add_argument("--variants", nargs="*")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("reextract", help="extract the SQL again from the stored raw outputs (no inference)")
    p.add_argument("--split", default="dev")
    p.add_argument("--models", nargs="*")
    p.add_argument("--variants", nargs="*")

    sub.add_parser("report", help="error analysis, summary tables and figures")

    args = parser.parse_args()
    config = load_config(args.config)
    if getattr(args, "split", None) is not None and args.split not in config.dataset.eval_splits:
        # train is the few-shot pool, not an evaluation split
        parser.error(f"--split must be one of {config.dataset.eval_splits}")
    _log_through_tqdm()

    if args.command == "build-db":
        from src.data.build_mimicsql_db import build_mimicsql_db

        build_mimicsql_db(config, force=args.force, keep_intermediate=args.keep_intermediate)
    elif args.command == "prepare":
        from src.data.convert import prepare

        prepare(config)
    elif args.command == "check-gold":
        from src.data.gold_check import check_gold

        check_gold(config)
    elif args.command == "infer":
        infer(config, args.split, args.models, args.variants, args.limit)
    elif args.command == "evaluate":
        evaluate(config, args.split, args.models, args.variants, args.overwrite)
    elif args.command == "reextract":
        reextract(config, args.split, args.models, args.variants)
    elif args.command == "report":
        from src.analysis.report import report

        report(config)


def existing_runs(config, split: str, models: list[str] | None, variants: list[str] | None):
    """(model, variant, run directory) for every selected run that has a raw.jsonl."""
    from src.inference.runner import run_dir

    model_cfgs = [config.model(m) for m in models] if models else config.models
    variant_cfgs = [config.variants[v] for v in (variants or default_variants(config, split))]
    for model in model_cfgs:
        for variant in variant_cfgs:
            path = run_dir(config, split, model, variant)
            if (path / "raw.jsonl").exists():
                yield model, variant, path


def evaluate(config, split: str, models: list[str] | None, variants: list[str] | None, overwrite: bool) -> None:
    from src.eval.mimic_core import MimicCore

    core = MimicCore(config)
    for model, variant, path in existing_runs(config, split, models, variants):
        core.evaluate_run(split, model.name, variant.name, path, overwrite=overwrite)


def reextract(config, split: str, models: list[str] | None, variants: list[str] | None) -> None:
    from loguru import logger

    from src.inference.runner import reextract_run

    for model, variant, path in existing_runs(config, split, models, variants):
        changed = reextract_run(config, split, path)
        logger.info(f"{model.name}/{variant.name}/{split}: {changed} predictions changed")


def default_variants(config, split: str) -> list[str]:
    if split == "test":
        if not config.selection.test_variant:
            raise SystemExit("selection.test_variant is not set; compare variants on dev first")
        return [config.selection.test_variant]
    return config.selection.dev_variants


def infer(config, split: str, models: list[str] | None, variants: list[str] | None, limit: int | None) -> None:
    from loguru import logger
    from tqdm import tqdm

    from src.inference.runner import InferenceRunner

    model_cfgs = [config.model(m) for m in models] if models else config.models
    variant_cfgs = [config.variants[v] for v in (variants or default_variants(config, split))]
    runners = {m.name: [InferenceRunner(config, m, v, split) for v in variant_cfgs] for m in model_cfgs}
    pending = {id(r): len(r.pending(limit)[1]) for rs in runners.values() for r in rs}
    total = sum(pending.values())
    logger.info(f"{split}: {len(model_cfgs)} models x {len(variant_cfgs)} variants, {total} answers to generate")

    with tqdm(total=total, desc="total", unit="q", position=0, dynamic_ncols=True) as overall:
        for i, model in enumerate(model_cfgs, 1):
            model_todo = sum(pending[id(r)] for r in runners[model.name])
            if model_todo == 0:
                logger.info(f"[model {i}/{len(model_cfgs)}] {model.name}: complete, skipped")
                continue
            logger.info(f"[model {i}/{len(model_cfgs)}] {model.name} ({model.tag}): {model_todo} answers to go")
            for j, runner in enumerate(runners[model.name], 1):
                runner.run(limit=limit, overall=overall,
                           label=f"[{i}/{len(model_cfgs)} | {j}/{len(variant_cfgs)}] ")
            runners[model.name][-1].unload()
            logger.info(f"[model {i}/{len(model_cfgs)}] {model.name}: unloaded")


if __name__ == "__main__":
    main()
