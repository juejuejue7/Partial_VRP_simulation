"""[A1] pytest 公共夹具 / sys.path 注入。

让 tests/ 下任意测试文件都能 `import msim.*`(项目无安装,靠根目录注入)。
本文件只做路径与小工具夹具,不含任何业务实现(A1 纪律:只写测试)。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
