"""Q64 —— `--splitter_yaml` 分支的单测。

覆盖三件事：
  1. 正常路径：yaml 的 train/val 与本地 log 取交集后逐位正确；
  2. 硬失败路径：yaml 要求的 log 在本地缺失时必须抛错，**不得静默缩小数据集**；
  3. 真实产物对拍：`gnn_scaling_L1.yaml` 与 S1 剔除后清单的 L1 逐位一致（这是 Q64 能成立的前提）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_log_disjoint import assign_logs_from_splitter  # noqa: E402

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SPLITTER_L1 = Path("/mnt/local/Xiao_Yi/external/planTF/config/splitter/gnn_scaling_L1.yaml")
MANIFEST = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42_filtered.json"


def _write_yaml(tmp: Path, train, val) -> Path:
    p = tmp / "fake_splitter.yaml"
    p.write_text(yaml.safe_dump({"log_splits": {"train": list(train), "val": list(val)}}))
    return p


class TestSplitterYaml(unittest.TestCase):

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_1_exact_split(self) -> None:
        """本地 log == yaml 全集 ⇒ 划分逐位等于 yaml。"""
        y = _write_yaml(self.tmp, ["a", "b", "c"], ["d"])
        tr, va = assign_logs_from_splitter(["a", "b", "c", "d"], y)
        self.assertEqual(tr, {"a", "b", "c"})
        self.assertEqual(va, {"d"})
        self.assertFalse(tr & va, "train/val 不得有交集")

    def test_2_missing_log_hard_fails(self) -> None:
        """🔴 splitter 要的 log 本地缺 pkl ⇒ 该级数据不完整，必须硬失败。"""
        y = _write_yaml(self.tmp, ["a", "b", "c"], ["d", "e"])
        with self.assertRaises(ValueError) as ctx:
            assign_logs_from_splitter(["a", "d"], y)
        self.assertIn("本地没有 pkl", str(ctx.exception))

    def test_3_extra_local_log_is_expected(self) -> None:
        """🔴 本地多出 log 是**预期**的：S4 一次产出 L3 全量，同一 data_dir 服务三级。

        这条曾经写反过（把 extra 当硬失败）—— 那会让 MAGNET L1 必然抛错，
        因为 data_dir 有 2292 个 log 而 L1 splitter 只有 236 个。
        """
        y = _write_yaml(self.tmp, ["a"], ["b"])
        tr, va = assign_logs_from_splitter(["a", "b", "EXTRA1", "EXTRA2"], y)
        self.assertEqual(tr, {"a"})
        self.assertEqual(va, {"b"})

    def test_4_yaml_train_val_overlap_fails(self) -> None:
        """yaml 自身 train/val 有交集 ⇒ 硬失败。"""
        y = _write_yaml(self.tmp, ["a", "b"], ["b"])
        with self.assertRaises(ValueError):
            assign_logs_from_splitter(["a", "b"], y)

    def test_5_empty_val_fails(self) -> None:
        """本地覆盖不到任何 val log ⇒ 硬失败（否则会静默训练无验证集）。"""
        y = _write_yaml(self.tmp, ["a", "b"], ["c"])
        with self.assertRaises(ValueError):
            assign_logs_from_splitter(["a"], y)

    def test_6_real_L1_matches_manifest(self) -> None:
        """🔴 真实产物对拍：yaml train∪val 必须逐位等于 S1 剔除后清单的 L1 logs。"""
        if not (SPLITTER_L1.exists() and MANIFEST.exists()):
            self.skipTest("真实产物不在，跳过")
        cfg = yaml.safe_load(SPLITTER_L1.read_text())
        tr, va = set(cfg["log_splits"]["train"]), set(cfg["log_splits"]["val"])
        man = set(json.loads(MANIFEST.read_text())["tiers"]["L1"]["logs"])
        self.assertEqual(tr | va, man, "yaml 与 S1 清单 L1 不一致 ⇒ Q64 的前提不成立")
        self.assertFalse(tr & va)
        self.assertEqual(len(tr), 212)
        self.assertEqual(len(va), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
