"""Checkpoint migration helpers for Agent 040's unchanged 104-input schema."""

from agent_code.Agent_039_compact_audit_ddqn_agent.checkpoint import (
    compact_agent038_checkpoint as _compact_agent038_checkpoint,
    policy_only_warm_start as _policy_only_warm_start,
)


def compact_agent038_checkpoint(checkpoint):
    result = _compact_agent038_checkpoint(checkpoint)
    result["migration"] = (
        "Agent 038 117-to-Agent 040 104-input compact schema; "
        "feature computation optimized without changing retained columns"
    )
    return result


def policy_only_warm_start(checkpoint, source=None):
    result = _policy_only_warm_start(checkpoint, source)
    result["warm_start"] = (
        "Agent 038 policy weights with Agent 040 input migration"
    )
    return result


__all__ = ["compact_agent038_checkpoint", "policy_only_warm_start"]
