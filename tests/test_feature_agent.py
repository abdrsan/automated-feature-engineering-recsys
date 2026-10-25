import polars as pl
import pytest
from src.feature_agent import propose_features, validate_llm_proposals
from src.feature_spec import FeatureSpec


def test_agent_is_deterministic_and_bounded():
    schema = {"user_id": pl.Int64, "item_id": pl.Int64, "rating": pl.Float64}
    current = [FeatureSpec(("user_id",), "positive", "count", 7)]
    importance = {current[0].name: 10}
    a = propose_features(schema, {"positive", "negative"}, current, [], importance, limit=2)
    b = propose_features(schema, {"positive", "negative"}, current, [], importance, limit=2)
    assert a == b and len(a) == 2
    assert all(s not in current for s in a)
    assert not propose_features(schema, {"positive", "negative"}, current,
                                 [{"proposed": [s.name for s in a]}], importance, limit=2)


def test_optional_llm_has_no_executable_payload():
    with pytest.raises((ValueError, TypeError)):
        validate_llm_proposals([{"keys": ["user_id"], "sql": "DELETE FROM data"}], {"user_id": pl.Int64}, {"positive"})
