from nl2sql360.core import Core
from nl2sql360.arguments import CoreArguments, DatasetArguments

core_args = CoreArguments()

core = Core(core_args)

# TODO: replace with corrept parameters
dataset_args = DatasetArguments(
    dataset_name="",
    dataset_dir=".",
    samples_file="",
    database_dir="",
    tables_file="",
    question_key="",
    sql_key="",
    db_id_key="",
    sql_complexity_key=None,
    database_domain_file=None
)

core.import_dataset(dataset_args)