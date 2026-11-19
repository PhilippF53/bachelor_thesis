"""Non-interactive deletion of NL2SQL360 tables.

`Core.delete_dataset_history` / `delete_evaluation_history` ask for confirmation via
input(), which fails in scripts. These helpers run the same SQL templates and also
forget the table classes, so the tables can be recreated in the same process.
"""

from nl2sql360.core import Core
from nl2sql360.database import (
    DELETE_DATASET_INFO,
    DELETE_DATASET_TABLE,
    DELETE_EVALUATION_TABLE,
    Base,
)
from sqlalchemy import text


def _forget(core: Core, table_name: str) -> None:
    core.models_dict.pop(table_name, None)
    if table_name in Base.metadata.tables:
        Base.metadata.remove(Base.metadata.tables[table_name])


def drop_evaluation(core: Core, dataset_name: str, eval_name: str) -> None:
    with core.engine.connect() as connection:
        connection.execute(text(DELETE_EVALUATION_TABLE.format(DATASET_NAME=dataset_name, EVAL_NAME=eval_name)))
        connection.commit()
    _forget(core, f"DATASET_{dataset_name}_EVALUATION_{eval_name}")


def drop_dataset(core: Core, dataset_name: str) -> None:
    """Drop the dataset table and its info row. Evaluations must be dropped first."""
    with core.engine.connect() as connection:
        connection.execute(text(DELETE_DATASET_TABLE.format(DATASET_NAME=dataset_name)))
        connection.execute(text(DELETE_DATASET_INFO.format(DATASET_NAME=dataset_name)))
        connection.commit()
    _forget(core, f"DATASET_{dataset_name}")
