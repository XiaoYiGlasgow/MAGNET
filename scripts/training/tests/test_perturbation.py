"""EgoPerturbation 正确性验证（阶段⑬ 强化版）。

覆盖：几何自洽、GT 全局不变性、route 通道免疫、零扰动恒等、
**gnn_x == 对变换后 x 的实际 subgraph pooling**（阶段⑬ 新增，因 heading wrap 非线性）、
固定 seed 可复现、val/test 不扰动。
"""
import sys
from pathlib import Path

import numpy as np
import torch

_GNN_ROOT = Path(__file__).resolve().parents[3]   # tests -> training -> scripts -> GNN_Trans
for _p in (str(_GNN_ROOT), str(_GNN_ROOT.parent), str(_GNN_ROOT / "scripts/training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perturbation import EgoPerturbation  # noqa: E402


class Dummy:
    pass


def make_data(seed=0, n_sub=3, per_sub=10):
    """构造 x（含 col9=subgraph_ID）与由其池化得到的 gnn_x，保证初始即自洽。"""
    g = torch.Generator().manual_seed(seed)
    d = Dummy()
    n = n_sub * per_sub
    x = torch.randn(n, 12, generator=g)
    ids = torch.arange(n) // per_sub
    x[:, 9] = ids.float()
    # heading 全部落在合法区间 [-pi, pi]（真实数据来自 arctan2），但**跨 ±pi 边界分布**：
    # 同一 subgraph 内既有 ≈+pi 又有 ≈-pi -> 均值与 wrap 不可交换，触发非线性（供判据 8/8b）。
    # 若造到 |h|>pi，wrap 本身就会改值，零扰动恒等（判据 1）会被误判为失败。
    side = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
    x[:, 4] = side * (3.05 + torch.rand(n, generator=g) * 0.08)
    x[:, 5] = side * (3.05 + torch.rand(n, generator=g) * 0.08)
    x[:, 6:9] = torch.tensor([0.3, 0.7, 1.0])          # route 三通道（per-lane 常量）
    d.x = x
    d.gnn_x = torch.stack([x[ids == k].mean(0) for k in range(n_sub)])
    d.y = torch.randn(60, 2, generator=g) * 0.5
    d.ego_history = torch.randn(20, 2, generator=g) * 0.3
    return d


def repool(x):
    ids = x[:, 9].round().long()
    return torch.stack([x[ids == k].mean(0) for k in sorted(set(ids.tolist()))])


def sample_like_call(p):
    """复现 __call__ 的 RNG 消耗顺序：先 random() 判概率，再 _sample()。"""
    p._rng.random()
    return p._sample()


ok = True
PARAMS = dict(max_lateral=2.0, max_longitudinal=1.0, max_yaw=0.3, prob=1.0)

# --- 1. 零扰动 = 恒等 ---
a, b = make_data(), make_data()
out = EgoPerturbation(max_lateral=0, max_longitudinal=0, max_yaw=0, prob=1.0, seed=1)(b)
same = (torch.allclose(a.x, out.x, atol=1e-6) and torch.allclose(a.y, out.y, atol=1e-5)
        and torch.allclose(a.gnn_x, out.gnn_x, atol=1e-5)
        and torch.allclose(a.ego_history, out.ego_history, atol=1e-6))
print(f"[1] 零扰动恒等（含 gnn_x）: {'✅' if same else '❌'}")
ok &= same

# --- 2. route 三通道 (col6-8) + col9-11 逐字节不变 ---
a, b = make_data(), make_data()
out = EgoPerturbation(**PARAMS, seed=7)(b)
imm = torch.equal(a.x[:, 6:9], out.x[:, 6:9])
untouched = torch.equal(a.x[:, 9:], out.x[:, 9:])
gimm = torch.allclose(a.gnn_x[:, 6:9], out.gnn_x[:, 6:9], atol=1e-6)
print(f"[2] x route 三通道免疫: {'✅' if imm else '❌'}  col9-11 不动: {'✅' if untouched else '❌'}"
      f"  gnn_x 三通道免疫: {'✅' if gimm else '❌'}")
ok &= imm and untouched and gimm

# --- 3. GT 全局不变性 ---
a, b = make_data(), make_data()
dx, dy, dt = sample_like_call(EgoPerturbation(**PARAMS, seed=3))
out = EgoPerturbation(**PARAMS, seed=3)(b)
C_old, C_new = torch.cumsum(a.y, 0), torch.cumsum(out.y, 0)
ct, st = np.cos(dt), np.sin(dt)
inv = torch.stack([C_new[:, 0] * ct - C_new[:, 1] * st + dx,
                   C_new[:, 0] * st + C_new[:, 1] * ct + dy], 1)
err = torch.max(torch.abs(inv - C_old)).item()
print(f"[3] GT 全局不变性(逆变换还原): {'✅' if err < 1e-4 else '❌'}  max_err={err:.2e}")
ok &= err < 1e-4

# --- 4. 节点位置变换正确 ---
a, b = make_data(), make_data()
dx, dy, dt = sample_like_call(EgoPerturbation(**PARAMS, seed=11))
out = EgoPerturbation(**PARAMS, seed=11)(b)
ct, st = np.cos(dt), np.sin(dt)
px, py = a.x[:, 0] - dx, a.x[:, 1] - dy
exp = torch.stack([px * ct + py * st, -px * st + py * ct], 1)
n_err = torch.max(torch.abs(exp - out.x[:, 0:2])).item()
print(f"[4] 节点位置变换正确: {'✅' if n_err < 1e-4 else '❌'}  max_err={n_err:.2e}")
ok &= n_err < 1e-4

# --- 5. heading wrap 到 [-pi, pi] ---
h_ok = bool((out.x[:, 4:6].abs() <= np.pi + 1e-6).all())
print(f"[5] heading wrap 到 [-pi,pi]: {'✅' if h_ok else '❌'}")
ok &= h_ok

# --- 6. ego_history 只旋转不平移（保模长）---
a, b = make_data(), make_data()
out = EgoPerturbation(**PARAMS, seed=5)(b)
r_ok = torch.allclose(torch.norm(a.ego_history, dim=1), torch.norm(out.ego_history, dim=1), atol=1e-5)
print(f"[6] ego_history 旋转保模长: {'✅' if r_ok else '❌'}")
ok &= r_ok

# --- 7. prob=0 完全恒等 ---
a, b = make_data(), make_data()
out = EgoPerturbation(prob=0.0, seed=2)(b)
p0 = torch.equal(a.x, out.x) and torch.equal(a.gnn_x, out.gnn_x)
print(f"[7] prob=0 完全恒等: {'✅' if p0 else '❌'}")
ok &= p0

# --- 8. ★ gnn_x == 对变换后 x 的实际 subgraph pooling（阶段⑬ 新增）---
a, b = make_data(), make_data()
out = EgoPerturbation(**PARAMS, seed=13)(b)
expect = repool(out.x)
g_err = torch.max(torch.abs(out.gnn_x - expect)).item()
g_ok = g_err < 1e-5
print(f"[8] gnn_x == repool(变换后 x): {'✅' if g_ok else '❌'}  max_err={g_err:.2e}")
ok &= g_ok

# --- 8b. 反证：独立变换 gnn_x（旧做法）在 heading 上不等价 ---
naive = a.gnn_x.clone()
dxn, dyn, dtn = sample_like_call(EgoPerturbation(**PARAMS, seed=13))
ctn, stn = np.cos(dtn), np.sin(dtn)
for cx, cy in [(0, 1), (2, 3)]:
    qx, qy = naive[:, cx] - dxn, naive[:, cy] - dyn
    naive[:, cx], naive[:, cy] = qx * ctn + qy * stn, -qx * stn + qy * ctn
for ch in (4, 5):
    naive[:, ch] = (naive[:, ch] - dtn + np.pi) % (2 * np.pi) - np.pi
head_gap = torch.max(torch.abs(naive[:, 4:6] - expect[:, 4:6])).item()
print(f"[8b] 旧做法(独立变换)在 heading 上的偏差: {head_gap:.3f} rad "
      f"-> {'✅ 证实必须重池化' if head_gap > 1e-3 else '⚠️ 本例未触发非线性'}")

# --- 9. 固定 seed 可复现 ---
o1 = EgoPerturbation(**PARAMS, seed=99)(make_data())
o2 = EgoPerturbation(**PARAMS, seed=99)(make_data())
rep = torch.equal(o1.x, o2.x) and torch.equal(o1.y, o2.y) and torch.equal(o1.gnn_x, o2.gnn_x)
print(f"[9] 固定 seed 可复现: {'✅' if rep else '❌'}")
ok &= rep

# --- 10. val/test 语义：不经 transform 即完全原样 ---
d, ref = make_data(), make_data()
none_ok = torch.equal(d.x, ref.x) and torch.equal(d.gnn_x, ref.gnn_x)
print(f"[10] val/test(transform=None) 不扰动: {'✅' if none_ok else '❌'}")
ok &= none_ok

print()
print("EgoPerturbation 验证:", "✅ 全部通过" if ok else "❌ 有失败")
sys.exit(0 if ok else 1)
