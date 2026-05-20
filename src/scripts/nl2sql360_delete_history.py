from nl2sql360.core import Core
from nl2sql360.arguments import CoreArguments, EvaluationArguments
from nl2sql360.filter import Filter, Scenario, Field, Operator

if __name__ == "__main__":
    core_args = CoreArguments()

    core = Core(core_args)
    
    # TODO: replace with corrept parameters
    core.delete_evaluation_history(
        dataset_name="",
        eval_name=""
    )
    
    core.delete_dataset_history(
        dataset_name="",
        delete_relavant_evaluations=True
    )