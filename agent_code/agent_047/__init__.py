"""Agent 050: self-contained Agent043 runtime and merged dependencies.

All custom dependency packages used by the Agent043 policy live directly in
this package and are imported with relative imports. Loading Agent 050 does
not modify the framework's ``agent_code`` package path.
"""
