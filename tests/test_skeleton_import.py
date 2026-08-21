"""[A0 波次0] 骨架级 import 闸门:全项目所有模块必须可 import(NIE 不在 import 期触发)。

可直接运行:  python tests/test_skeleton_import.py
也兼容 pytest: pytest tests/test_skeleton_import.py
"""
from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = [
    "msim",
    # contracts
    "msim.contracts", "msim.contracts.types", "msim.contracts.config",
    "msim.contracts.geometry", "msim.contracts.physics",
    "msim.contracts.task", "msim.contracts.mdp",
    "msim.contracts.env_static",
    # L0 config
    "msim.config", "msim.config.config",
    # L0/L1 geometry
    "msim.geometry", "msim.geometry.grid", "msim.geometry.window",
    # L1 env_static
    "msim.env_static", "msim.env_static.field", "msim.env_static.leader_path",
    # L2a physics
    "msim.physics", "msim.physics.auv_dynamics", "msim.physics.leader",
    "msim.physics.follower", "msim.physics.rollout",
    # L2b task
    "msim.task", "msim.task.corridor", "msim.task.value",
    "msim.task.belief", "msim.task.energy_balance",
    # L3 mdp
    "msim.mdp", "msim.mdp.commitment", "msim.mdp.state",
    "msim.mdp.action", "msim.mdp.coop_survey_env",
    # L4 policy
    "msim.policy", "msim.policy.network", "msim.policy.train_ppo",
    # L5 baselines / eval
    "msim.baselines", "msim.baselines.ortools_vrp", "msim.baselines.exact_milp",
    "msim.baselines.greedy", "msim.eval", "msim.eval.metrics", "msim.eval.runner",
]


def test_all_modules_import() -> None:
    failed = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            failed.append((m, repr(e)))
    assert not failed, "import 失败:\n" + "\n".join(f"  {m}: {e}" for m, e in failed)


def test_null_updater_is_open_loop() -> None:
    """开环性最小证明:NullUpdater.update 后 field 逐位不变(无需重型依赖)。"""
    import numpy as np

    from msim.physics.follower import CameraSensorModel
    from msim.task.belief import NullUpdater

    field = np.array([[0.1, 0.9], [0.5, 0.3]], dtype=np.float64)
    before = field.copy()
    NullUpdater().update(field, {(0, 0), (1, 1)}, CameraSensorModel())
    assert np.array_equal(field, before), "NullUpdater 破坏了开环性(field 被改写)"


if __name__ == "__main__":
    ok = True
    for _m in MODULES:
        try:
            importlib.import_module(_m)
            print("OK  ", _m)
        except Exception as _e:  # noqa: BLE001
            ok = False
            print("FAIL", _m, "->", repr(_e))
    try:
        test_null_updater_is_open_loop()
        print("OK   open-loop(NullUpdater)")
    except Exception as _e:  # noqa: BLE001
        ok = False
        print("FAIL open-loop ->", repr(_e))
    print("\n== smoke import:", "ALL PASS ==" if ok else "FAILURES ==")
    sys.exit(0 if ok else 1)
