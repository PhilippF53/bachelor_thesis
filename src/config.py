from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Paths:
    source_db: Path
    mimicsql_dir: Path
    treqs_dir: Path
    derived_dir: Path
    nl2sql360_dir: Path
    runs_dir: Path
    results_dir: Path
    docs_dir: Path

    @property
    def database_dir(self) -> Path:
        return self.derived_dir / "database"

    def db_path(self, db_id: str) -> Path:
        return self.database_dir / db_id / f"{db_id}.sqlite"

    def split_results_dir(self, split: str) -> Path:
        return self.results_dir / split

    @property
    def tables_file(self) -> Path:
        return self.derived_dir / "tables.json"

    def split_file(self, dataset: str, split: str) -> Path:
        return self.derived_dir / f"{dataset}_{split}.json"


@dataclass
class DatasetConfig:
    name: str
    db_id: str
    question_key: str
    splits: list[str]
    eval_splits: list[str]


@dataclass
class ExecutionConfig:
    timeout_s: float = 30


@dataclass
class EvaluationConfig:
    # repetitions per correct prediction for the VES/R-VES time ratio (BIRD: 100); 0 = off
    ves_iterations: int = 100


@dataclass
class ReportConfig:
    # models with known training-data contamination; left out of the cross-model
    # comparison "solved by how many models"
    contaminated_models: list[str] = field(default_factory=list)


@dataclass
class InferenceConfig:
    seed: int = 42
    temperature: float = 0
    num_ctx: int = 8192
    num_predict: int = 1024
    keep_alive: str = "30m"


@dataclass
class VariantConfig:
    name: str
    k: int = 0
    strategy: str | None = None
    # DB values that occur in the question are listed below the schema (None = off, so
    # manifests written before this field existed still match)
    values: bool | None = None


@dataclass
class ModelConfig:
    name: str
    tag: str
    format: str
    think: bool | str | None = None
    num_ctx: int | None = None
    num_predict: int | None = None
    extra_options: dict | None = None  # further Ollama options, e.g. workarounds for one model

    def options(self, inference: InferenceConfig) -> dict:
        return {
            "temperature": inference.temperature,
            "seed": inference.seed,
            "num_ctx": self.num_ctx or inference.num_ctx,
            "num_predict": self.num_predict or inference.num_predict,
            **(self.extra_options or {}),
        }


@dataclass
class SelectionConfig:
    dev_variants: list[str]
    test_variant: str | None = None


@dataclass
class Config:
    paths: Paths
    dataset: DatasetConfig
    execution: ExecutionConfig
    evaluation: EvaluationConfig
    report: ReportConfig
    inference: InferenceConfig
    variants: dict[str, VariantConfig]
    selection: SelectionConfig
    models: list[ModelConfig]
    raw: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.paths.db_path(self.dataset.db_id)

    def model(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"unknown model: {name}")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        paths=Paths(**{k: Path(v) for k, v in raw["paths"].items()}),
        dataset=DatasetConfig(**raw["dataset"]),
        execution=ExecutionConfig(**raw.get("execution", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        report=ReportConfig(**raw.get("report", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        variants={name: VariantConfig(name=name, **v) for name, v in raw.get("variants", {}).items()},
        selection=SelectionConfig(**raw["selection"]),
        models=[ModelConfig(**m) for m in raw.get("models", [])],
        raw=raw,
    )
