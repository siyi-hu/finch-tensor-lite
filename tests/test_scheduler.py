from finch.autoschedule import propagate_map_queries
from finch.finch_logic import *


def test_propagate_map_queries_simple():
    plan = Plan(
        (
            Query(Alias("A10"), Aggregate(Immediate("+"), Immediate(0), Immediate("[1,2,3]"), ())),
            Query(Alias("A11"), Alias("A10")),
            Produces((Alias("11"),)),
        )
    )
    expected = Plan(
        (
            Query(Alias("A11"), MapJoin(Immediate("+"), (Immediate(0), Immediate("[1,2,3]")))),
            Produces((Alias("11"),)),
        )
    )

    result = propagate_map_queries(plan)
    assert result == expected
