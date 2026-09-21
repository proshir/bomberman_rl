"""Agent 050: self-contained tournament package of the Agent043 policy.

The final-project submission is a single agent directory. Agent043 was
developed on top of several earlier experimental agents, so the required
runtime modules are vendored below ``_vendor/agent_code``. Extend the
framework package path before those historical absolute imports resolve.
"""

from pathlib import Path

import agent_code as _framework_agent_code


_VENDORED_ROOT = str(Path(__file__).resolve().parent / "_vendor" / "agent_code")
if _VENDORED_ROOT not in _framework_agent_code.__path__:
    # ``agent_code`` is a namespace package in the supplied framework.
    # ``_NamespacePath`` exposes its mutable search list as ``_path``.
    _framework_agent_code.__path__._path.insert(0, _VENDORED_ROOT)
