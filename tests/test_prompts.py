from pathlib import Path

from src.config import load_config
from src.data.convert import normalize_literals
from src.inference.fewshot import FewShotSelector
from src.inference.formats import RENDERERS, render
from src.inference.prompts import Example, build_content

FORMATS = ("chat", "sqlcoder_v1", "sqlcoder_v2", "duckdb_nsql")
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
    assert set(FORMATS) == set(RENDERERS)
    for fmt in FORMATS:
        text = render(fmt, content).as_text()
        assert "CREATE TABLE LAB" in text
        assert "how many lab tests?" in text
        assert "Question: q1\nSQL: SELECT 1" in text
        assert "SQLite" in text


def test_sqlcoder_prompts_follow_the_layout_of_their_checkpoint():
    content = build_content("CREATE TABLE LAB (LABEL TEXT);", "how many lab tests?", [Example("q1", "SELECT 1")])
    v1 = render("sqlcoder_v1", content).prompt
    v2 = render("sqlcoder_v2", content).prompt
    # defog/sqlcoder: Instructions / Input / Response
    assert v1.startswith("### Instructions:\nYour task is to convert a question into a SQL query")
    assert "\n\n### Input:\nGenerate a SQL query that answers the question `how many lab tests?`.\n" in v1
    assert "This query will run on a database whose schema is represented in this string:\nCREATE TABLE LAB" in v1
    assert v1.endswith("### Response:\nBased on your instructions, here is the SQL query I have generated to "
                       "answer the question `how many lab tests?`:\n```sql\n")
    # defog/sqlcoder-7b and sqlcoder2: Task / Database Schema / SQL
    assert v2.startswith("### Task\nGenerate a SQLite query to answer the following question:\n`how many lab tests?`\n")
    assert ("\n\n### Database Schema\nThe query will run on a database with the following schema:\n"
            "CREATE TABLE LAB") in v2
    assert v2.endswith("### SQL\nGiven the database schema, here is the SQL query that answers "
                       "`how many lab tests?`:\n```sql\n")
    for prompt in (v1, v2):
        # tags of the later sqlcoder-7b-2 format, which these checkpoints do not know
        assert "[QUESTION]" not in prompt and "[SQL]" not in prompt
        assert prompt.count("```") == 1  # only the opened block at the end


def test_prompts_ending_with_an_opened_code_block_have_the_fence_as_stop_sequence():
    """Without the stop sequence the model writes on after the closing fence, and the
    extraction may take that text for the answer."""
    content = build_content("CREATE TABLE LAB (LABEL TEXT);", "q", [])
    checked = set()
    for path in sorted(Path("config").glob("*.yaml")):
        config = load_config(path)
        for model in config.models:
            prompt = render(model.format, content)
            if prompt.mode == "raw" and prompt.prompt.endswith("```sql\n"):
                assert "```" in model.options(config.inference).get("stop", []), f"{path.name}: {model.name}"
                checked.add(model.name)
    assert {"sqlcoder-7b", "sqlcoder-15b"} <= checked


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
    assert prompt_template(ModelConfig(name="m", tag="m:1", format="sqlcoder_v2"), variant, "S") != \
        prompt_template(model, variant, "S")
    prompts.RULES.append("extra rule")
    try:
        assert prompt_template(model, variant, "CREATE TABLE LAB (LABEL TEXT);") != before
    finally:
        prompts.RULES.pop()


def test_value_hints_only_change_prompts_that_use_them():
    plain = build_content("CREATE TABLE LAB (LABEL TEXT);", "q", [])
    hinted = build_content("CREATE TABLE LAB (LABEL TEXT);", "q", [], ["LAB.LABEL = 'Glucose'"])
    assert plain.schema_text == plain.schema
    for fmt in FORMATS:
        assert "Values mentioned" not in render(fmt, plain).as_text()
        assert "-- LAB.LABEL = 'Glucose'" in render(fmt, hinted).as_text()


def test_value_matcher_uses_database_spelling(tmp_path):
    import sqlite3

    from src.inference.values import ValueMatcher

    # synthetic toy database with the MIMICSQL table names, no MIMIC data
    db = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE DIAGNOSES (ICD9_CODE TEXT, SHORT_TITLE TEXT)")
    conn.executemany("INSERT INTO DIAGNOSES VALUES (?, ?)",
                     [("4019", "Hypertension NOS"), ("V5861", "Hypertension"), ("2724", "Lipidemia O'Neil")])
    conn.execute("CREATE TABLE DEMOGRAPHIC (GENDER TEXT, DOB TEXT)")
    conn.executemany("INSERT INTO DEMOGRAPHIC VALUES (?, ?)", [("F", "2100-01-01"), ("M", "2100-01-02")])
    conn.commit()
    conn.close()

    hints = ValueMatcher(db).hints("how many f patients have hypertension nos and code v5861 or 4019 born 2100-01-01?")
    assert "DIAGNOSES.SHORT_TITLE = 'Hypertension NOS'" in hints
    assert "DIAGNOSES.SHORT_TITLE = 'Hypertension'" not in hints  # contained in the longer match
    assert "DIAGNOSES.ICD9_CODE = 'V5861'" in hints
    assert "DEMOGRAPHIC.GENDER = 'F'" in hints
    assert not any("4019" in h or "DOB" in h for h in hints)  # numbers and skipped columns
    assert ValueMatcher(db).hints("lipidemia o'neil") == ["DIAGNOSES.SHORT_TITLE = 'Lipidemia O''Neil'"]


def test_prompt_template_marks_value_hints():
    from src.config import ModelConfig, VariantConfig
    from src.inference.runner import prompt_template

    model = ModelConfig(name="m", tag="m:1", format="chat")
    plain = prompt_template(model, VariantConfig(name="a", k=3, strategy="similar"), "S")
    hinted = prompt_template(model, VariantConfig(name="b", k=3, strategy="similar", values=True), "S")
    assert "<VALUE_HINT>" in hinted and "<VALUE_HINT>" not in plain
