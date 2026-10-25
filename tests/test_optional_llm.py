import polars as pl
import pytest
from src.feature_agent import propose_features


def test_provider_hook_is_optional_and_allowlisted():
    schema = {"user_id": pl.Int64, "item_id": pl.Int64}
    seen = []
    def provider(context):
        seen.append(context)
        return [{"keys": ["user_id"], "event": "positive", "aggregation": "count", "window_days": 10}]
    proposals = propose_features(schema, {"positive", "negative"}, [], [], {}, llm_provider=provider)
    assert len(proposals) == 1 and proposals[0].window_days == 10
    assert seen[0]["max_proposals"] == 12
    with pytest.raises(ValueError):
        propose_features(schema, {"positive"}, [], [], {}, llm_provider=lambda context:
                         [{"keys": ["item_id", "user_id"], "window_days": 10}])
