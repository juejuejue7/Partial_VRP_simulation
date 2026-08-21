"""pytest 路径引导:让 `vrpsim`(本目录下)与 `msim`(仓库根)都能被 import。"""
from __future__ import annotations

import os
import sys

SIM_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SIM_ROOT)

for p in (SIM_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
