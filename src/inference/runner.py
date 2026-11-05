"""Run one (model, variant, split) inference job with resume support.

Layout: runs/<split>/<model>/<variant>/
  manifest.json    config snapshot, timestamps, model tag + digest, Ollama version, git commit
  raw.jsonl        one record per question (prompt, raw output, thinking, extracted SQL, timings)
  predictions.sql  one SQL per line in dataset order (written when the run is complete)
"""

import contextlib
import fcntl
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config import Config, ModelConfig, VariantConfig
from src.inference.extract import extract_sql
from src.inference.fewshot import FewShotSelector
from src.inference.formats import RenderedPrompt, render
from src.inference.prompts import Example, build_content, build_schema_text
from src.inference.values import ValueMatcher
from src.services.ollama_service import LLMResponse, OllamaService

NO_SQL = "NO_SQL_EXTRACTED"


class ContextOverflowError(RuntimeError):
    pass


def run_dir(config: Config, split: str, model: ModelConfig, variant: VariantConfig) -> Path:
    return config.paths.runs_dir / split / model.name / variant.name


def load_samples(config: Config, split: str) -> list[dict]:
    return json.loads(config.paths.split_file(config.dataset.name, split).read_text())


def load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["key"]] = rec
    return records


@contextlib.contextmanager
def run_lock(path: Path):
    """Exclusive lock on a run directory, so two processes never write the same run
    (duplicate records, distorted latencies)."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{path} is used by another process") from None
        yield


def write_predictions(path: Path, samples: list[dict], records: dict[str, dict]) -> None:
    """predictions.sql, one SQL per line in dataset order; only once every sample is answered."""
    if all(s["key"] in records for s in samples):
        preds = [records[s["key"]]["pred_sql"] or NO_SQL for s in samples]
        (path / "predictions.sql").write_text("\n".join(preds) + "\n")


def reextract_run(config: Config, split: str, path: Path) -> int:
    """Run the SQL extraction again on the stored raw outputs (no inference) and update
    pred_sql/extract_method. Returns the number of changed records."""
    raw_path = path / "raw.jsonl"
    with run_lock(path):
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = 0
        for rec in records:
            sql, method = extract_sql(rec["content"])
            if (sql, method) != (rec["pred_sql"], rec["extract_method"]):
                rec["pred_sql"], rec["extract_method"] = sql, method
                changed += 1
        if changed:
            tmp = raw_path.with_name(raw_path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            tmp.replace(raw_path)
            write_predictions(path, load_samples(config, split), {r["key"]: r for r in records})
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest.setdefault("reextracted", []).append(
                    {"at": datetime.now().isoformat(timespec="seconds"), "changed": changed})
                manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return changed


def _git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def prompt_template(model: ModelConfig, variant: VariantConfig, schema: str) -> str:
    """The rendered prompt with placeholders: instruction, rules, schema and format exactly
    as the model sees them, independent of the individual question and examples."""
    examples = [Example("<EXAMPLE_QUESTION>", "<EXAMPLE_SQL>")] * variant.k
    value_hints = ["<VALUE_HINT>"] if variant.values else []
    return render(model.format, build_content(schema, "<QUESTION>", examples, value_hints)).as_text()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _without_none(value):
    """Config fields added later default to None; a manifest written before the field
    existed lacks the key. Both mean the same, so None-valued keys are ignored."""
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if v is not None}
    return value


def _settings(config: Config, model: ModelConfig, variant: VariantConfig, schema: str) -> dict:
    """Everything that must stay identical when a run is resumed."""
    return {
        "model": asdict(model),
        "options": model.options(config.inference),
        "variant": asdict(variant),
        "schema_sha256": _sha256(schema),
        "prompt_template_sha256": _sha256(prompt_template(model, variant, schema)),
    }


class InferenceRunner:
    def __init__(self, config: Config, model: ModelConfig, variant: VariantConfig, split: str) -> None:
        self.config = config
        self.model = model
        self.variant = variant
        self.split = split
        self.dir = run_dir(config, split, model, variant)
        self.options = model.options(config.inference)
        self.llm = OllamaService(model.tag, self.options, think=model.think,
                                 keep_alive=config.inference.keep_alive)
        self.schema = build_schema_text(config.db_path)
        self.selector = None
        if variant.k > 0:
            self.selector = FewShotSelector(load_samples(config, "train"), variant.strategy,
                                            variant.k, config.inference.seed)
        self.values = ValueMatcher(config.db_path) if variant.values else None

    def render(self, sample: dict) -> tuple[RenderedPrompt, list]:
        examples = self.selector.select(sample) if self.selector else []
        hints = self.values.hints(sample["question"]) if self.values else []
        content = build_content(self.schema, sample["question"], examples, hints)
        return render(self.model.format, content), examples

    def call(self, prompt: RenderedPrompt) -> LLMResponse:
        if prompt.mode == "raw":
            return self.llm.generate_raw(prompt.prompt)
        return self.llm.chat(prompt.messages)

    def _check_context(self, resp: LLMResponse) -> dict:
        num_ctx, num_predict = self.options["num_ctx"], self.options["num_predict"]
        prompt_tokens = resp.prompt_tokens or 0
        if prompt_tokens + num_predict > num_ctx:
            raise ContextOverflowError(
                f"prompt ({prompt_tokens} tokens) + num_predict ({num_predict}) exceeds num_ctx ({num_ctx})"
            )
        return {"output_truncated": resp.done_reason == "length"}

    def _stored_prompts_match(self) -> bool:
        """True if the current rendering reproduces every stored prompt exactly."""
        samples = {s["key"]: s for s in load_samples(self.config, self.split)}
        for rec in load_records(self.dir / "raw.jsonl").values():
            if rec["key"] not in samples or self.render(samples[rec["key"]])[0].as_text() != rec["prompt"]:
                return False
        return True

    def _resumed_manifest(self, settings: dict) -> dict | None:
        """Existing manifest if this run may be resumed; raises if anything changed."""
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())
        # the template hash covers instruction, rules, schema and format, but not the
        # question texts and few-shot examples; the stored prompts cover everything
        if not self._stored_prompts_match():
            raise RuntimeError(f"{manifest_path}: prompt text changed since the run started; "
                               "archive or delete the old run")
        old = manifest["settings"]
        old.setdefault("prompt_template_sha256", settings["prompt_template_sha256"])  # manifests before the hash
        changed = sorted(k for k in settings if _without_none(old.get(k)) != _without_none(settings[k]))
        if changed:
            raise RuntimeError(f"{manifest_path}: {', '.join(changed)} changed since the run "
                               "started; archive or delete the old run")
        if manifest["model_info"]["digest"] != self.llm.model_info()["digest"]:
            raise RuntimeError(f"{manifest_path}: model digest changed since the run started")
        return manifest

    def _write_manifest(self, manifest: dict | None, settings: dict, gpu: dict) -> None:
        manifest_path = self.dir / "manifest.json"
        if manifest is not None:
            manifest.setdefault("resumed", []).append(datetime.now().isoformat(timespec="seconds"))
        else:
            manifest = {
                "created": datetime.now().isoformat(timespec="seconds"),
                "split": self.split,
                "config": self.config.raw,
                "settings": settings,
                "model_info": self.llm.model_info(),
                "ollama_version": OllamaService.server_version(),
                "git": _git_state(),
                "gpu": gpu,
            }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    def _gpu_placement(self) -> dict:
        for m in self.llm.client.ps().models:
            if m.model == self.model.tag:
                return {"size": m.size, "size_vram": m.size_vram, "fully_on_gpu": m.size_vram >= m.size}
        return {}

    def pending(self, limit: int | None = None) -> tuple[list[dict], list[dict]]:
        """(all samples of this run, samples without a stored answer)."""
        samples = load_samples(self.config, self.split)
        if limit:
            samples = samples[:limit]
        done = load_records(self.dir / "raw.jsonl")
        return samples, [s for s in samples if s["key"] not in done]

    def run(self, limit: int | None = None, overall: tqdm | None = None, label: str = "") -> Path:
        with run_lock(self.dir):
            return self._run(limit, overall, label)

    def _run(self, limit: int | None, overall: tqdm | None, label: str) -> Path:
        samples, todo = self.pending(limit)  # read under the lock
        raw_path = self.dir / "raw.jsonl"
        name = f"{label}{self.model.name}/{self.variant.name}/{self.split}"
        logger.info(f"{name}: {len(samples) - len(todo)} done, {len(todo)} to go")

        if todo:
            # warm-up with the longest prompt: loads the model (load time stays out of the
            # latency numbers) and checks that no prompt gets truncated by num_ctx
            settings = _settings(self.config, self.model, self.variant, self.schema)
            manifest = self._resumed_manifest(settings)  # fails before loading the model
            logger.info(f"{name}: warm-up ({self.model.tag}, loads the model if needed)")
            start = time.perf_counter()
            longest = max(samples, key=lambda s: len(self.render(s)[0].as_text()))
            warmup = self.call(self.render(longest)[0])
            self._check_context(warmup)
            gpu = self._gpu_placement()
            logger.info(f"{name}: ready after {time.perf_counter() - start:.1f}s "
                        f"(load {warmup.load_duration_s or 0:.1f}s, longest prompt {warmup.prompt_tokens} "
                        f"tokens, num_ctx {self.options['num_ctx']}, on GPU: {gpu.get('fully_on_gpu')})")
            if gpu and not gpu["fully_on_gpu"]:
                logger.warning(f"{self.model.tag} is partially offloaded to CPU: {gpu}")
            self._write_manifest(manifest, settings, gpu)

        latencies, out_tokens, no_sql, truncated = [], [], 0, 0
        bar = tqdm(todo, desc=f"{label}{self.model.name}/{self.variant.name}", position=1, leave=False,
                   dynamic_ncols=True)
        with open(raw_path, "a", encoding="utf-8") as f:
            for sample in bar:
                prompt, examples = self.render(sample)
                resp = self.call(prompt)
                flags = self._check_context(resp)
                sql, method = extract_sql(resp.content)
                record = {
                    "key": sample["key"],
                    "id": sample["id"],
                    "prompt": prompt.as_text(),
                    "example_questions": [e.question for e in examples],
                    **({"n_value_hints": len(self.values.hints(sample["question"]))} if self.values else {}),
                    **resp.to_dict(),
                    "pred_sql": sql,
                    "extract_method": method,
                    **flags,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                latencies.append(resp.wall_s)
                out_tokens.append(resp.output_tokens or 0)
                no_sql += sql is None
                truncated += flags["output_truncated"]
                bar.set_postfix(lat=f"{sum(latencies) / len(latencies):.1f}s",
                                tok=f"{sum(out_tokens) / len(out_tokens):.0f}", no_sql=no_sql)
                if overall is not None:
                    overall.update(1)
        bar.close()
        if latencies:
            logger.info(f"{name}: finished {len(latencies)} answers, mean latency "
                        f"{sum(latencies) / len(latencies):.2f}s, mean output tokens "
                        f"{sum(out_tokens) / len(out_tokens):.0f}, no SQL {no_sql}, truncated {truncated}")

        write_predictions(self.dir, samples, load_records(raw_path))
        return self.dir

    def unload(self) -> None:
        self.llm.unload()
