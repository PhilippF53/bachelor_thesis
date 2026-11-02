"""Model-independent prompt content: instruction, schema, examples, question."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

INSTRUCTION = (
    "Given the schema of a clinical SQLite database (derived from MIMIC-III) and a question, "
    "write one SQL query that answers the question."
)
RULES = [
    "The target dialect is SQLite. Do not use syntax or functions specific to DuckDB, PostgreSQL or MySQL.",
    "Use only the tables and columns listed in the schema.",
    "Return only the SQL query, without explanation.",
]

# Short column descriptions following the MIMICSQL/TREQS table definitions.
COLUMN_DESCRIPTIONS = {
    "DEMOGRAPHIC": {
        "SUBJECT_ID": "patient id",
        "HADM_ID": "hospital admission id",
        "NAME": "patient name",
        "MARITAL_STATUS": "marital status",
        "AGE": "age at admission in years",
        "DOB": "date of birth",
        "GENDER": "gender",
        "LANGUAGE": "language",
        "RELIGION": "religion",
        "ADMISSION_TYPE": "type of admission",
        "DAYS_STAY": "length of hospital stay in days",
        "INSURANCE": "insurance type",
        "ETHNICITY": "ethnicity",
        "EXPIRE_FLAG": "1 if the patient died, else 0",
        "ADMISSION_LOCATION": "admission location",
        "DISCHARGE_LOCATION": "discharge location",
        "DIAGNOSIS": "primary diagnosis at admission (free text)",
        "DOD": "date of death",
        "DOB_YEAR": "year of birth",
        "DOD_YEAR": "year of death",
        "ADMITTIME": "admission time",
        "DISCHTIME": "discharge time",
        "ADMITYEAR": "year of admission",
    },
    "DIAGNOSES": {
        "SUBJECT_ID": "patient id",
        "HADM_ID": "hospital admission id",
        "ICD9_CODE": "ICD-9 diagnosis code",
        "SHORT_TITLE": "short diagnosis title",
        "LONG_TITLE": "long diagnosis title",
    },
    "PROCEDURES": {
        "SUBJECT_ID": "patient id",
        "HADM_ID": "hospital admission id",
        "ICD9_CODE": "ICD-9 procedure code",
        "SHORT_TITLE": "short procedure title",
        "LONG_TITLE": "long procedure title",
    },
    "PRESCRIPTIONS": {
        "SUBJECT_ID": "patient id",
        "HADM_ID": "hospital admission id",
        "ICUSTAY_ID": "ICU stay id",
        "DRUG_TYPE": "drug type",
        "DRUG": "drug name",
        "FORMULARY_DRUG_CD": "drug code",
        "ROUTE": "route of administration",
        "DRUG_DOSE": "drug dose with unit",
    },
    "LAB": {
        "SUBJECT_ID": "patient id",
        "HADM_ID": "hospital admission id",
        "ITEMID": "lab item id",
        "CHARTTIME": "time of the lab test",
        "FLAG": "abnormality flag of the result",
        "VALUE_UNIT": "result value with unit",
        "LABEL": "lab test name",
        "FLUID": "fluid of the lab test",
        "CATEGORY": "lab test category",
    },
}


@dataclass
class Example:
    question: str
    sql: str


@dataclass
class PromptContent:
    instruction: str
    rules: list[str]
    schema: str
    question: str
    examples: list[Example] = field(default_factory=list)

    @property
    def rules_text(self) -> str:
        return "\n".join(f"- {r}" for r in self.rules)

    @property
    def examples_text(self) -> str:
        return "\n\n".join(f"Question: {e.question}\nSQL: {e.sql}" for e in self.examples)


def build_schema_text(db_path: Path) -> str:
    """CREATE TABLE statements with one comment per column; no table contents."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    blocks = []
    for table, descriptions in COLUMN_DESCRIPTIONS.items():
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        lines = []
        for i, col in enumerate(cols):
            name, col_type = col[1], col[2] or "TEXT"
            comma = "," if i < len(cols) - 1 else ""
            desc = descriptions.get(name)
            lines.append(f"  {name} {col_type}{comma}" + (f" -- {desc}" if desc else ""))
        blocks.append(f"CREATE TABLE {table} (\n" + "\n".join(lines) + "\n);")
    conn.close()
    blocks.append("-- All tables can be joined on HADM_ID (hospital admission id).")
    return "\n\n".join(blocks)


def build_content(schema: str, question: str, examples: list[Example]) -> PromptContent:
    return PromptContent(INSTRUCTION, RULES, schema, question, examples)
