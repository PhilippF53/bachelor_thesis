from src.data.convert import normalize_literals
from src.inference.fewshot import FewShotSelector
from src.inference.formats import render
from src.inference.prompts import Example, build_content

TRAIN = [
    {"key": f"k{i}", "question": q, "sql_norm": f"SELECT {i}"}
    for i, q in enumerate([
        "how many patients are female?",
        "what is the drug code of aspirin?",
        "count the number of patients with diagnosis short title hypertension",
        "what is the average age of male patients?",
        "how many patients were admitted in 2150?",
    ])
]
SAMPLE = {"key": "dev1", "question": "count the number of patients with diagnosis short title diabetes"}


def test_random_selection_is_deterministic_per_key():
    a = FewShotSelector(TRAIN, "random", 3, seed=42).select(SAMPLE)
    b = FewShotSelector(list(TRAIN), "random", 3, seed=42).select(SAMPLE)
    assert a == b and len(a) == 3


def test_similar_selection_puts_best_match_last():
    examples = FewShotSelector(TRAIN, "similar", 2, seed=42).select(SAMPLE)
    assert examples[-1].question == TRAIN[2]["question"]


def test_formats_share_content_and_name_sqlite():
    content = build_content("CREATE TABLE LAB (LABEL TEXT);", "how many lab tests?", [Example("q1", "SELECT 1")])
    for fmt in ("chat", "sqlcoder", "duckdb_nsql"):
        text = render(fmt, content).as_text()
        assert "CREATE TABLE LAB" in text
        assert "how many lab tests?" in text
        assert "Question: q1\nSQL: SELECT 1" in text
        assert "SQLite" in text
    assert render("sqlcoder", content).prompt.endswith("[SQL]\n")


def test_normalize_literals_keeps_identifiers():
    sql = 'SELECT T."A" FROM T WHERE T."B" = "x" AND T."C" < "40"'
    assert normalize_literals(sql) == "SELECT T.\"A\" FROM T WHERE T.\"B\" = 'x' AND T.\"C\" < '40'"


def test_prompt_template_tracks_prompt_text_only():
    from src.config import ModelConfig, VariantConfig
    from src.inference import prompts
    from src.inference.runner import prompt_template

    model = ModelConfig(name="m", tag="m:1", format="chat")
    variant = VariantConfig(name="few", k=3, strategy="random")
    before = prompt_template(model, variant, "CREATE TABLE LAB (LABEL TEXT);")
    assert before.count("<EXAMPLE_QUESTION>") == 3
    assert prompt_template(ModelConfig(name="m", tag="m:1", format="sqlcoder"), variant, "S") != \
        prompt_template(model, variant, "S")
    prompts.RULES.append("extra rule")
    try:
        assert prompt_template(model, variant, "CREATE TABLE LAB (LABEL TEXT);") != before
    finally:
        prompts.RULES.pop()
