"""Model-specific rendering of the same PromptContent.

`chat` goes through the model's chat template. `sqlcoder` and `duckdb_nsql` are sent
raw in the format the models were fine-tuned on.
"""

from dataclasses import dataclass

from src.inference.prompts import PromptContent


@dataclass
class RenderedPrompt:
    mode: str  # "chat" or "raw"
    messages: list[dict] | None = None
    prompt: str | None = None

    def as_text(self) -> str:
        if self.mode == "raw":
            return self.prompt
        return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in self.messages)


def render_chat(c: PromptContent) -> RenderedPrompt:
    system = f"{c.instruction}\n\nRules:\n{c.rules_text}"
    user = f"### Database schema\n{c.schema_text}\n\n"
    if c.examples:
        user += f"### Examples\n{c.examples_text}\n\n"
    user += f"### Question\nQuestion: {c.question}\nSQL:"
    return RenderedPrompt("chat", messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


def render_sqlcoder(c: PromptContent) -> RenderedPrompt:
    prompt = (
        f"### Task\nGenerate a SQLite query to answer [QUESTION]{c.question}[/QUESTION]\n\n"
        f"### Instructions\n{c.instruction}\n{c.rules_text}\n\n"
        f"### Database Schema\nThe query will run on a database with the following schema:\n{c.schema_text}\n\n"
    )
    if c.examples:
        prompt += f"### Examples\n{c.examples_text}\n\n"
    prompt += (
        f"### Answer\nGiven the database schema, here is the SQL query that "
        f"[QUESTION]{c.question}[/QUESTION]\n[SQL]\n"
    )
    return RenderedPrompt("raw", prompt=prompt)


def render_duckdb_nsql(c: PromptContent) -> RenderedPrompt:
    # Same section layout as the duckdb-nsql template, with SQLite named as target dialect.
    prompt = (
        "### Instruction:\nYour task is to generate valid SQLite SQL (not DuckDB SQL) to answer "
        "the following question, given a SQLite database schema.\n"
        f"{c.instruction}\n{c.rules_text}\n\n"
        f"### Input:\nHere is the database schema that the SQL query will run on:\n{c.schema_text}\n\n"
    )
    if c.examples:
        prompt += f"Examples:\n{c.examples_text}\n\n"
    prompt += f"### Question:\n{c.question}\n\n### Response (use SQLite syntax):\n"
    return RenderedPrompt("raw", prompt=prompt)


RENDERERS = {
    "chat": render_chat,
    "sqlcoder": render_sqlcoder,
    "duckdb_nsql": render_duckdb_nsql,
}


def render(fmt: str, content: PromptContent) -> RenderedPrompt:
    return RENDERERS[fmt](content)
