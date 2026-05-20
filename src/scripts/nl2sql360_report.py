from nl2sql360.core import Core
from nl2sql360.arguments import CoreArguments, EvaluationArguments
from nl2sql360.filter import Filter, Scenario, Field, Operator

if __name__ == "__main__":
    core_args = CoreArguments()

    core = Core(core_args)
    
    SUBQUERY_FILTER = Filter(
        name="subquery",
        field=Field.SUBQUERY,
        operator=Operator.GT,
        value=0
    )
    
    BI_SCENARIO = Scenario(
        name="BI",
        filters=[Filter('agg', Field.AGGREGATION, Operator.GT, 0), Filter('join', Field.JOIN, Operator.GT, 0)]
    )
