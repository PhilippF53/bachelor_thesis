"""DB values mentioned in a question, in database spelling (value hints for the prompt).

The MIMICSQL template questions are lower case, while the database stores values in
their original case and `=` on text is case-sensitive in SQLite. The matcher finds the
distinct values of the text columns that occur literally (ignoring case, on word
boundaries) in the question. Values stay in memory and only go into the prompt.
"""

import re
import sqlite3
from pathlib import Path

from src.inference.prompts import COLUMN_DESCRIPTIONS

# times and measurement values: unique per row, no help for the conditions in MIMICSQL
SKIP_COLUMNS = {"DOB", "DOD", "ADMITTIME", "DISCHTIME", "CHARTTIME", "VALUE_UNIT"}


def _is_number(value: str) -> bool:
    return value.replace(".", "", 1).isdigit()


class ValueMatcher:
    def __init__(self, db_path: Path) -> None:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
        self._values = []  # (table, column, value, pattern)
        for table in COLUMN_DESCRIPTIONS:
            for col in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
                name, col_type = col[1], (col[2] or "").upper()
                if col_type != "TEXT" or name in SKIP_COLUMNS:
                    continue
                rows = conn.execute(f'SELECT DISTINCT "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL '
                                    f'ORDER BY "{name}"')
                for (value,) in rows:
                    value = str(value).strip()
                    # numbers have no spelling problem
                    if value and not _is_number(value):
                        pattern = re.compile(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)")
                        self._values.append((table, name, value, pattern))
        conn.close()
        self._cache: dict[str, list[str]] = {}

    def hints(self, question: str) -> list[str]:
        """`TABLE.COLUMN = 'value'` for every matching value; a value contained in a longer
        match of the same column is dropped (e.g. 'hypertension' inside 'hypertension nos')."""
        if question not in self._cache:
            q = question.lower()
            hits = [(t, c, v) for t, c, v, pattern in self._values if pattern.search(q)]
            hits = [(t, c, v) for t, c, v in hits
                    if not any(c2 == c and t2 == t and v.lower() != v2.lower() and v.lower() in v2.lower()
                               for t2, c2, v2 in hits)]
            self._cache[question] = [f"{t}.{c} = '{v.replace(chr(39), chr(39) * 2)}'" for t, c, v in hits]
        return self._cache[question]
