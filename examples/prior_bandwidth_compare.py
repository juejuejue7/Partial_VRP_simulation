"""[A0 诊断] 先验场集中度 vs KDE 带宽 σ —— 同一组 ground-truth 目标点,扫不同 bandwidth。

用途:让人类直观选定先验"不确定范围"的合适带宽。σ 越小→先验越集中(峰更尖、晕开更小)。
不改任何 builder 文件,只 import 已验收实现。
运行:D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/prior_bandwidth_compare.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclasses import replace

from msim.contracts.config import SimConfig, WorldConfig
from msim.env_static.field import field_from_targets
from msim.eval.runner import default_instance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

cfg = replace(SimConfig(), world=WorldConfig(x_max_m=60.0, y_max_m=30.0, res_m=0.5), seed=0)
world = cfg.world
inst = default_instance(cfg, n_targets=6)          # 与 acceptance_demo Part3 完全相同的目标点
targets = inst.targets
tg = np.asarray([np.asarray(t, float) for t in targets])

bandwidths = [12.0, 6.0, 3.0, 1.5]                 # 12 = 当前默认(=intra_spread);依次更集中

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ext = [0, world.x_max_m, 0, world.y_max_m]
fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
for ax, bw in zip(axes.ravel(), bandwidths):
    f = field_from_targets(targets, world, bandwidth_m=bw)
    # 量化"不确定范围":先验 > 0.5 的面积占比
    frac = float((f > 0.5).mean()) * 100.0
    im = ax.imshow(f.T, origin="lower", extent=ext, aspect="auto",
                   cmap="viridis", vmin=0.0, vmax=1.0)
    ax.scatter(tg[:, 0], tg[:, 1], s=55, facecolors="red", edgecolors="white",
               linewidths=1.0, zorder=5)
    tag = " (current default)" if bw == 12.0 else ""
    ax.set_title(f"sigma = {bw:.1f} m{tag}   |  field>0.5 area = {frac:.0f}%", fontsize=10)
    ax.set_xlabel("North x (m)", fontsize=8)
    ax.set_ylabel("East y (m)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
fig.suptitle("Prior-field concentration vs KDE bandwidth (sigma), same ground-truth targets\n"
             "smaller sigma = tighter prior = smaller uncertainty region", fontsize=11)
p = os.path.join(OUT, "prior_bandwidth_compare.png")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(p, dpi=110)
plt.close(fig)
print("targets:", [tuple(np.round(np.asarray(t, float), 1)) for t in targets])
for bw in bandwidths:
    f = field_from_targets(targets, world, bandwidth_m=bw)
    print(f"  sigma={bw:>5.1f} m  ->  field>0.5 area = {float((f>0.5).mean())*100:5.1f}%  "
          f"field>0.2 area = {float((f>0.2).mean())*100:5.1f}%")
print("saved:", p)
