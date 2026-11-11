import pytest

from src.inference.extract import extract_sql

Q = 'SELECT COUNT ( DISTINCT DEMOGRAPHIC."SUBJECT_ID" ) FROM DEMOGRAPHIC WHERE DEMOGRAPHIC."GENDER" = \'F\''


@pytest.mark.parametrize("response, method", [
    (Q, "plain"),
    (Q + ";", "plain"),
    (f"```sql\n{Q};\n```", "code_block"),
    (f"Here is the query:\n```sqlite\n{Q}\n```\nIt counts female patients.", "code_block"),
    (f"```\n{Q}\n```", "code_block"),
    (f"<think>Maybe SELECT * FROM LAB?</think>\n```sql\n{Q}\n```", "code_block"),
    (f"To answer this, we count patients.\n\n{Q};\n\nThis query filters by gender.", "plain"),
    (f"SQL: {Q}", "plain"),
    (f"{Q}\n[/SQL]", "plain"),
    (f"[SQL] {Q} [/SQL]", "sql_tag"),
    # sqlcoder-7b with a prompt that ends with "[SQL]" (archived runs): another opening tag follows
    (f" {Q};\n[SQL]", "plain"),
    (f" {Q}\n[SQL]", "plain"),
    # SQLCoder with the opened ```sql block and "```" as stop sequence: only the query arrives
    (f"{Q};\n", "plain"),
    (f" {Q}\n", "plain"),
    (f"```sql\n{Q}", "code_block"),
    # stray closing fence after plain SQL (seen with qwen3-coder)
    (f"{Q}\n```", "plain"),
    (f"```sql\n-- SELECT all female patients\n{Q}\n```", "code_block"),
    (f"```sql\n/* female patients */\n{Q}\n```", "code_block"),
    (f"The query with a filter on gender:\n{Q}\n\nWith this query you get the count.", "plain"),
])
def test_extracts_query(response, method):
    sql, got_method = extract_sql(response)
    assert sql == Q
    assert got_method == method


def test_last_sql_block_wins():
    response = f"```sql\nSELECT 1\n```\nCorrected:\n```sql\n{Q}\n```"
    assert extract_sql(response)[0] == Q


def test_multiline_is_collapsed_but_literals_kept():
    response = "```sql\nSELECT *\n  FROM LAB\n  WHERE LAB.\"LABEL\" = 'a;  b'\n```"
    assert extract_sql(response)[0] == "SELECT * FROM LAB WHERE LAB.\"LABEL\" = 'a;  b'"


def test_inline_comment_does_not_swallow_the_rest():
    response = "```sql\nSELECT * FROM LAB -- all lab tests\nWHERE LAB.\"LABEL\" = 'a -- b'\n```"
    assert extract_sql(response)[0] == "SELECT * FROM LAB WHERE LAB.\"LABEL\" = 'a -- b'"


@pytest.mark.parametrize("response", ["", "I cannot answer that.", "<think>SELECT 1 FROM LAB", None])
def test_no_sql(response):
    assert extract_sql(response)[0] is None


def test_with_clause():
    response = "WITH x AS (SELECT 1) SELECT * FROM x"
    assert extract_sql(response)[0] == response
