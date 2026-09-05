"""Phase-1A-R Step 2: thin live Webots ODE research adapter.

Plant motion is produced exclusively by Webots ODE via Supervisor.step().
Python never integrates x/y/yaw for propulsion.
"""

from .env import LiveWebotsEnv
from .types import ControllerObservation, LiveObservation, PrivilegedEvalState

__all__ = [
    "LiveWebotsEnv",
    "LiveObservation",
    "ControllerObservation",
    "PrivilegedEvalState",
]
