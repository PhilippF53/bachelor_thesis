"""Extract a single SQL statement from a model response."""

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```[ \t]*[a-zA-Z]*[ \t]*\n?(.*?)```", re.DOTALL)
_OPEN_CODE_BLOCK = re.compile(r"```[ \t]*[a-zA-Z]*[ \t]*\n?(.*)$", re.DOTALL)
_SQL_TAG = re.compile(r"\[SQL\](.*?)(?:\[/SQL\]|$)", re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)
# in free text the prose words "select"/"with" are lower case, SQL keywords usually not
_SQL_START_UPPER = re.compile(r"\b(SELECT|WITH)\b")
_BLANK_LINE = re.compile(r"\n\s*\n")
_LABEL_PREFIX = re.compile(r"^\s*(?:sql(?:ite)?\s*(?:query)?\s*:)\s*", re.IGNORECASE)


def _strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    # unterminated thinking (e.g. hit num_predict): nothing after it is an answer
    if re.search(r"<think>", text, re.IGNORECASE):
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.replace("</think>", "")


def _strip_comments(sql: str) -> str:
    """Remove -- and /* */ comments outside of quotes; line breaks are kept."""
    out, quote, i = [], None, 0
    while i < len(sql):
        ch = sql[i]
        if quote is None and sql.startswith("--", i):
            end = sql.find("\n", i)
            i = len(sql) if end == -1 else end
            continue
        if quote is None and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = len(sql) if end == -1 else end + 2
            out.append(" ")
            continue
        if quote is None and ch in ("'", '"', "`"):
            quote = ch
        elif ch == quote:
            quote = None
        out.append(ch)
        i += 1
    return "".join(out)


def _first_statement(sql: str) -> str:
    """Cut at the first ';' outside of quotes and collapse whitespace outside of quotes.

    Comments are removed first: once the lines are joined, a '--' comment would swallow
    the rest of the query.
    """
    out, quote = [], None
    for ch in _strip_comments(sql):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
            out.append(ch)
        elif ch == ";":
            break
        elif ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out).strip()


def _from_code_blocks(text: str) -> str | None:
    """The last code block with SQL, also an unterminated one; None if no block has SQL."""
    blocks = [body for body in _CODE_BLOCK.findall(text) if _SQL_START.search(body)]
    if blocks:
        return blocks[-1]
    m = _OPEN_CODE_BLOCK.search(text)
    return m.group(1) if m and _SQL_START.search(m.group(1)) else None


def _sql_start(text: str, prose: bool) -> int | None:
    m = (prose and _SQL_START_UPPER.search(text)) or _SQL_START.search(text)
    return None if m is None else m.start()


def extract_sql(response: str) -> tuple[str | None, str]:
    """Return (sql, method). sql is None if nothing usable was found.

    Sources in order: code block with SQL, [SQL] tag, free text. Plain SQL followed by
    a stray closing fence therefore counts as free text.
    """
    text = _strip_think(response or "")
    if not text.strip():
        return None, "empty"

    if "```" in text and (block := _from_code_blocks(text)) is not None:
        candidate, method = _strip_comments(block), "code_block"
    elif m := _SQL_TAG.search(text):
        candidate, method = _strip_comments(m.group(1)), "sql_tag"
    else:
        # free text may use "--" as a dash; comments are removed only after the SQL start
        candidate, method = text.replace("```", ""), "plain"

    candidate = _LABEL_PREFIX.sub("", candidate.replace("[/SQL]", ""))
    start = _sql_start(candidate, prose=method == "plain")
    if start is None:
        return None, "no_sql"
    candidate = candidate[start:]
    if method == "plain":
        # free text: stop at a blank line (explanation follows)
        candidate = _BLANK_LINE.split(candidate, maxsplit=1)[0]
    sql = _first_statement(candidate)
    return (sql, method) if sql else (None, "no_sql")
