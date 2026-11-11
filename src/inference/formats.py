"""Model-specific rendering of the same PromptContent.

`chat` goes through the model's chat template. The other formats are sent raw in the
layout the checkpoint was fine-tuned on:

- `sqlcoder_v1`: defog/sqlcoder (StarCoder, 15B), the checkpoint behind the Ollama tag
  `sqlcoder:15b` according to its library page
- `sqlcoder_v2`: defog/sqlcoder-7b (Mistral, Ollama tag `sqlcoder:7b`) and defog/sqlcoder2
- `duckdb_nsql`: DuckDB-NSQL-7B

Both SQLCoder prompts end with an opened ```sql block, and the answer ends at the closing
fence. Models with these formats therefore need "```" as stop sequence (`extra_options`
in the config), as in Defog's inference.py; otherwise the model writes on after the fence
and the extraction may pick up that text.
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


def render_sqlcoder_v1(c: PromptContent) -> RenderedPrompt:
    # Layout of github.com/defog-ai/sqlcoder, prompt.md of 2023-08-18 (commit 2313019), with
    # SQLite instead of Postgres, our instruction and rules, and the examples after the schema.
    prompt = (
        "### Instructions:\nYour task is to convert a question into a SQL query, given a SQLite "
        f"database schema.\n{c.instruction}\nAdhere to these rules:\n{c.rules_text}\n\n"
        f"### Input:\nGenerate a SQL query that answers the question `{c.question}`.\n"
        f"This query will run on a database whose schema is represented in this string:\n{c.schema_text}\n\n"
    )
    if c.examples:
        prompt += f"Examples:\n{c.examples_text}\n\n"
    prompt += (
        "### Response:\nBased on your instructions, here is the SQL query I have generated to answer "
        f"the question `{c.question}`:\n```sql\n"
    )
    return RenderedPrompt("raw", prompt=prompt)


def render_sqlcoder_v2(c: PromptContent) -> RenderedPrompt:
    # Layout of github.com/defog-ai/sqlcoder, prompt.md of 2023-10-04 (commit 36d7795), with
    # SQLite named in the task and the sections Instructions and Examples added.
    prompt = (
        f"### Task\nGenerate a SQLite query to answer the following question:\n`{c.question}`\n\n"
        f"### Instructions\n{c.instruction}\n{c.rules_text}\n\n"
        f"### Database Schema\nThe query will run on a database with the following schema:\n{c.schema_text}\n\n"
    )
    if c.examples:
        prompt += f"### Examples\n{c.examples_text}\n\n"
    prompt += f"### SQL\nGiven the database schema, here is the SQL query that answers `{c.question}`:\n```sql\n"
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
    "sqlcoder_v1": render_sqlcoder_v1,
    "sqlcoder_v2": render_sqlcoder_v2,
    "duckdb_nsql": render_duckdb_nsql,
}


def render(fmt: str, content: PromptContent) -> RenderedPrompt:
    return RENDERERS[fmt](content)
