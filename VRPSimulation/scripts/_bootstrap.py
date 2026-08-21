"""脚本共用的 sys.path 引导 + 控制台编码兜底(Windows GBK 控制台遇到日文会炸)。"""
from __future__ import annotations

import os
import sys

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_ROOT)

for _p in (SIM_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
