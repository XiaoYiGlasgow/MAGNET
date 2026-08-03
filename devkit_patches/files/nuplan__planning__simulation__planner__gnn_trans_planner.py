"""GNN+Transformer MAGNET planner（路线B，VT-2026-04191 阶段2/3 重写）。

在线特征构建**完全复用训练同一链路**：
  encoding_vectornet_features(E0 契约参数) → build_graph_data_from_feature_df（与 GraphDataset.process 同一 helper）
契约对齐 E0：历史 20帧/2s、未来 60帧/6s、agents=20、radius=50、地图 LANE-only（max 40×50）、
单一 default edge builder、EGO_HISTORY[20,2] 局部逐帧增量。

模型输出 60×2 局部增量 → cumsum → 旋转回全局 → 纯参考轨迹；运动学可行性由 simulation 层
two_stage_controller（tracker=lqr/ilqr + kinematic_bicycle_model）强制，planner **不**在内部搓运动学。

删除全部回退（匀速基线+0.02×output / 随机初始化 / Linear(1,120) / mock 特征 / 第二套 edge builder /
静止后备 / 异常吞噬 / debug 写文件）——加载或推理失败即 raise，闭环评测计为失败、不静默替换。
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    Point2D, StateSE2, StateVector2D, TimePoint,
)
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

# GNN_Trans 源码（config / src.shared.utils）+ parent models（TransformerPredictor E0 生效副本）上 sys.path
_DEVKIT_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit"
_GNN_TRANS_ROOT = f"{_DEVKIT_ROOT}/GNN_Trans"
_MODELS_DIR = f"{_DEVKIT_ROOT}/models"
for _p in (_GNN_TRANS_ROOT, _MODELS_DIR, _DEVKIT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HEADING_EPS = 1e-3  # 位移过小则沿用上一朝向（避免 atan2 抖动）
_GNN_CONFIG = None    # GNN_Trans/config.py（按文件路径显式加载，避开 run_simulation 里 'config' 名冲突）


class _CollapseProbe:
    """Q46 —— **逐仿真步**记录输出头是否塌陷。默认不启用；启用时对模型输出**零影响**。

    为什么要挂在 planner 里而不是事后关联
    ------------------------------------
    干预审计的塌陷读数在 val 3000 token 上测，官方闭环在 heldout 256 场景上跑，
    2026-08-01 实测两者 **token 交集 0、log 交集 0** ⇒ 事后逐 token 关联**建不起来**。
    而 §7 硬门要求「塌陷必须与闭环失效逐 token 关联后才能写进论文」。

    挂进 planner 则一次闭环就同时拿到两侧，而且**比事后关联更强**：
    闭环里的塌陷发生在 ego 已经漂移出数据分布的状态上，离线 val 特征根本表示不了那些状态。

    为什么对输出零影响
    ------------------
    PyTorch 的 forward hook **返回 None 时不修改模块输出**；本类的 hook 只做
    `detach()` + 统计，不参与梯度也不改张量。单测 `test_collapse_probe.py` 判据 A
    对同一批输入逐位比对「开/关探针」的轨迹，要求 `max|Δ| == 0`。

    :param out_path: JSONL 落盘路径。每步一行，含 `iteration` / `collapsed` / `zero_frac`。
    """

    def __init__(self, out_path: Path) -> None:
        self.path = out_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 🔴 截断而不是追加：一个 scenario 一个 planner 实例、一个文件。
        #   若用 "a"，重跑同一个 group 会**静默累积** —— `n_steps` 翻倍、
        #   `first_collapse_step` 取到上一轮的值，而文件看上去完全正常。
        #   截断使重跑幂等；覆盖时留一条 warning，免得无声丢掉上一轮数据。
        if self.path.exists():
            logger.warning("[CollapseProbe] 覆盖已存在的探针文件（重跑）: %s", self.path)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._last: "dict | None" = None
        self._handle = None

    def attach(self, model) -> None:
        """挂到 `output_head` 的**最后一个 ReLU** 上（与离线审计探针同一挂点）。

        该 ReLU 恒零 ⇒ 末层 `Linear(128,120)` 输入是零向量、输出退化为**纯 bias**
        ⇒ 该步预测与 ego / context / GNN 三条分支的输入全部无关。
        挂不上就 raise —— 探针是显式要求的，静默失效等于伪造证据。
        """
        head = getattr(model, "output_head", None)
        relus = [m for m in head if isinstance(m, torch.nn.ReLU)] if head is not None else []
        if not relus:
            raise RuntimeError("[CollapseProbe] output_head 里找不到 ReLU，探针无法挂载")
        self._handle = relus[-1].register_forward_hook(self._capture)

    def _capture(self, _module, _inp, out) -> None:
        o = out.detach()
        self._last = {
            "collapsed": bool(float(o.abs().max()) == 0.0),
            "zero_frac": round(float((o == 0).float().mean()), 6),
        }

    def record(self, scenario_id: dict, iteration: int) -> None:
        """把本步的读数落盘。未捕获到激活时也要写一行，**不允许静默丢步**。"""
        row = {"iteration": iteration, **scenario_id}
        row.update(self._last if self._last is not None else {"collapsed": None, "zero_frac": None})
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._last = None


def _load_gnn_config():
    """按文件路径显式加载 GNN_Trans/config.py（E0 契约常量单一源）。

    run_simulation 进程里 sys.modules 已缓存别处的 'config' 模块，裸 `import config` 会取到错模块，
    故用 importlib 直接从已知文件路径加载，保证拿到 GNN_Trans 的 config（NUM_AGENTS/MAP_*/RADIUS 等）。
    """
    global _GNN_CONFIG
    if _GNN_CONFIG is None:
        spec = importlib.util.spec_from_file_location("gnn_trans_config", f"{_GNN_TRANS_ROOT}/config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GNN_CONFIG = module
    return _GNN_CONFIG


class GNNTransPlanner(AbstractPlanner):
    """基于 MAGNET TransformerPredictor 的 imitation planner（路线B，官方闭环评测）。"""

    #: 让 `build_planners` 注入 `scenario`（官方机制，`LogFuturePlanner` 同款）。
    #: 只取 token / log_name / scenario_type 三个字符串给塌陷探针做 join key，
    #: **不持有 scenario 对象本身**（避免 planner 被序列化时拖着整个场景）。
    #: ⚠️ 唯一受影响的其他调用点是 `nuplan/submission/challenge_servicers.py:98`
    #: （它传 scenario=None），那条路径本项目从不使用。
    requires_scenario: bool = True

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        horizon_seconds: float = 6.0,
        sampling_time: float = 0.1,
        adjacency_config: str = "default",
        route_mode: str = "none",
        strict_route_mode: bool = True,
        gnn_pooling_fix: bool = False,
        collapse_probe_dir: "str | None" = None,
        planning_layer: "str | None" = None,
        reference_source: "str | None" = None,
        scenario=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # ── C4（R4-1 规划形式化）：推理侧规划层。**默认关闭**，关闭时本 planner 的输出
        #    与加入这两个参数之前**逐位一致**（单测判据 [L]）。
        #    两种开法，yaml 参数优先；都没给就读环境变量，再没有就是关闭：
        #      ① `planner.gnn_trans_planner.planning_layer=full|no_progress|...`
        #      ② 环境变量 `MAGNET_PLANNING_LAYER=<preset>`
        #    走环境变量的理由同 Q46 探针（见 `collapse_probe_dir` 的注释）：不改
        #    `run_closed_loop.sh` ⇒ harness sha 不变 ⇒ 不触发规则基线四格重跑。
        from src.planning_layer import config as _pl_config  # noqa: E402

        self._planning_layer_cfg = (
            _pl_config.preset(planning_layer) if planning_layer else _pl_config.preset_from_env()
        )
        # 上游来源：`model`（默认，MAGNET）/ `cv`（常速外推）。
        # 🔴 `cv` 是 C4 的**诚实性支点**（设计文档 §4.1）：若「CV+层」≈「MAGNET+层」，
        #    结论只能写「在该规划形式化下学习组件无可测量的额外贡献」。
        self._reference_source = (
            reference_source or os.environ.get("MAGNET_REFERENCE_SOURCE") or "model"
        ).strip()
        if self._reference_source not in ("model", "cv"):
            raise ValueError(
                f"[GNNTransPlanner] 非法 reference_source={self._reference_source!r}，只允许 model/cv"
            )
        self._drivable = None          # 懒建：只有开了规划层才碰地图
        self._route_centerline = None
        # Q46 塌陷探针：默认关闭，行为与加入本参数之前**逐字节一致**。
        # 两种开法，yaml 参数优先；都没给就是关闭：
        #   ① `planner.gnn_trans_planner.collapse_probe_dir=<dir>`（hydra override）
        #   ② 环境变量 `MAGNET_COLLAPSE_PROBE_DIR=<dir>`
        # 🔴 为什么要留②：`run_closed_loop.sh` 没有通用 override 透传，而给它加一个 `--collapse-probe`
        #    会改动 harness ⇒ 按既定规矩要重跑规则基线四格、且 provenance 里的 harness sha 会分叉。
        #    走环境变量则**探针组与 primary 组的 provenance 只差一个 env var**，比两个 harness 版本更干净。
        self._collapse_probe_dir = collapse_probe_dir or os.environ.get("MAGNET_COLLAPSE_PROBE_DIR") or None
        self._probe: "_CollapseProbe | None" = None
        self._scenario_id = {
            "token": getattr(scenario, "token", None),
            "log_name": getattr(scenario, "log_name", None),
            "scenario_type": getattr(scenario, "scenario_type", None),
        }
        if route_mode not in ("none", "three_channel"):
            raise ValueError(f"[GNNTransPlanner] 非法 route_mode={route_mode!r}，只允许 none/three_channel")
        self._route_mode = route_mode
        self._strict_route_mode = strict_route_mode
        # D0#7：与训练侧同名 flag。batch=1 时 pooling ON≡OFF（global_mean_pool 单图 ≡ mean(0)），
        # 故对 planner 部署函数无影响；保留该参数是为了 fix=True 训练的 ckpt 能对齐加载路径。
        self._gnn_pooling_fix = bool(gnn_pooling_fix)
        self._checkpoint_path = checkpoint_path
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._horizon_seconds = horizon_seconds
        self._sampling_time = sampling_time
        self._num_output_points = int(round(horizon_seconds / sampling_time))  # 6s/0.1s=60
        self._adjacency_config = adjacency_config

        self.future_trajectory_sampling = TrajectorySampling(
            num_poses=self._num_output_points,
            time_horizon=horizon_seconds,
            interval_length=sampling_time,
        )
        self._model = None
        self._map_api = None
        self._route_roadblock_ids: List[str] = []  # 由 initialize 注入（D1-2 route 契约）
        self._diag_logged = False  # 首个 planner 调用打印一次 history 帧数（契约核对）
        logger.info(
            "[GNNTransPlanner] device=%s horizon=%.1fs dt=%.2fs points=%d adjacency=%s route_mode=%s "
            "planning_layer=%s reference_source=%s",
            self._device, horizon_seconds, sampling_time, self._num_output_points,
            adjacency_config, route_mode,
            self._planning_layer_cfg.provenance(), self._reference_source,
        )
        if self._planning_layer_cfg.enabled or self._reference_source != "model":
            # 规划层 / CV 上游的时间栅格必须与 planner 完全一致，否则候选与输出点位错开一步。
            # 不静默对齐 —— 失配即 raise（D0 无回退原则）。
            if (self._planning_layer_cfg.horizon != self._num_output_points
                    or abs(self._planning_layer_cfg.dt - sampling_time) > 1e-9):
                raise ValueError(
                    f"[GNNTransPlanner] 规划层时间栅格失配："
                    f"层 horizon={self._planning_layer_cfg.horizon} dt={self._planning_layer_cfg.dt}，"
                    f"planner points={self._num_output_points} dt={sampling_time}"
                )

    # ---- checkpoint 与 route_mode 一致性（旧 ckpt 从未见过 route 通道，喂进去会失配）----
    def _check_route_mode_consistency(self) -> None:
        """比对 checkpoint 训练时的 route_mode 与当前 planner 配置。

        训练脚本会在 checkpoint 同目录写 sidecar `train_meta.json`（含 route_mode 等）。
        旧的 E1 checkpoint 没有该 sidecar，按 `route_mode="none"` 处理 —— 这正是实测结论：
        把 route 三通道喂给从未见过它的 E1 checkpoint，会造成预测漂移
        （10 个 E0 样本：ADE 2.45 m / FDE 7.10 m / 最大终点偏移 20.74 m），
        即阶段5 的旧基线在无该保护时**无法复现**。
        """
        meta_path = os.path.join(os.path.dirname(self._checkpoint_path), "train_meta.json")
        ckpt_mode = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    ckpt_mode = json.load(f).get("route_mode")
            except (OSError, ValueError) as e:
                logger.warning("[GNNTransPlanner] 读取 %s 失败: %s", meta_path, e)
        if ckpt_mode is None:
            ckpt_mode = "none"
            logger.warning(
                "[GNNTransPlanner] checkpoint 无 train_meta.json sidecar，按 route_mode='none' 处理: %s",
                self._checkpoint_path,
            )
        if ckpt_mode != self._route_mode:
            msg = (f"[GNNTransPlanner] route_mode 失配：checkpoint 训练用 '{ckpt_mode}'，"
                   f"planner 配置 '{self._route_mode}'。喂入模型未见过的输入会使结果不可解释。"
                   f"（如确需如此，设 strict_route_mode=False 降级为 warning）")
            if self._strict_route_mode:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info("[GNNTransPlanner] route_mode 一致性校验通过: %s", ckpt_mode)

        self._check_adjacency_consistency(meta_path)

    def _check_adjacency_consistency(self, meta_path: str) -> None:
        """比对 checkpoint 训练时的 `dataset_config` 与 planner 的 `adjacency_config`。

        🔴 为什么必须查（2026-08-01 C3 起跑前发现）：C2 的三格用 `--dataset_config nearby_only`
        训练，而本 planner 的 `adjacency_config` 默认是 `"default"`（完全图），
        且此前**没有任何校验** ⇒ 会静默地拿完全图喂给只见过 30 m 近邻图的模型。

        对 C2 尤其致命：C2 的靶点正是「完全图上 GCN 一层即过平滑」，
        用完全图部署等于把要验证的修复在部署侧原样撤销 —— 闭环读数会变成无意义的负结果。

        与 route_mode 同样受 `strict_route_mode` 控制（同一个「部署契约」开关）。
        老 checkpoint 的 sidecar 里若无 `dataset_config`，按 `"default"` 处理（那正是它们的训练配置）。
        """
        ckpt_cfg = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    ckpt_cfg = json.load(f).get("dataset_config")
            except (OSError, ValueError) as e:
                logger.warning("[GNNTransPlanner] 读取 %s 失败: %s", meta_path, e)
        if ckpt_cfg is None:
            ckpt_cfg = "default"
            logger.warning("[GNNTransPlanner] checkpoint sidecar 无 dataset_config，按 'default' 处理")
        if ckpt_cfg != self._adjacency_config:
            msg = (f"[GNNTransPlanner] 图拓扑失配：checkpoint 训练用 dataset_config='{ckpt_cfg}'，"
                   f"planner 配置 adjacency_config='{self._adjacency_config}'。"
                   f"覆盖方式：planner.gnn_trans_planner.adjacency_config={ckpt_cfg}"
                   f"（如确需如此，设 strict_route_mode=False 降级为 warning）")
            if self._strict_route_mode:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info("[GNNTransPlanner] 图拓扑一致性校验通过: %s", ckpt_cfg)

    # ---- 模型加载（无回退：失败即 raise）----
    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not os.path.exists(self._checkpoint_path):
            raise FileNotFoundError(f"[GNNTransPlanner] checkpoint 不存在: {self._checkpoint_path}")

        # G13′-A / S4-b：走 bn_calibration 的统一加载入口，原始 ckpt 与 calibrated ckpt 都能加载。
        # calibrated 权重比原始多 ego_history_encoder.1.running_{mean,var}/num_batches_tracked 三个键，
        # 该入口据此自动决定是否先换 BN 再 load；旧的 bare state_dict 路径行为不变（无这三个键即原样加载）。
        # ⚠️ 不静默兜底：加载失败一律 raise（D0 无回退原则）。
        from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402

        model = load_model_for_inference(
            Path(self._checkpoint_path),
            gnn_pooling_fix=self._gnn_pooling_fix,
            device=self._device,
        )
        self._model = model
        bn = model.ego_history_encoder[1]
        logger.info("[GNNTransPlanner] 已加载 checkpoint: %s (calibrated=%s, gnn_pooling_fix=%s)",
                    self._checkpoint_path,
                    bool(getattr(bn, "track_running_stats", False)), self._gnn_pooling_fix)
        self._check_route_mode_consistency()

        # Q46：探针只在显式给了目录时挂载；一个 scenario 一个 JSONL，文件名即 join key
        if self._collapse_probe_dir:
            tok = self._scenario_id.get("token") or "unknown_token"
            log = self._scenario_id.get("log_name") or "unknown_log"
            self._probe = _CollapseProbe(Path(self._collapse_probe_dir) / f"{log}__{tok}.jsonl")
            self._probe.attach(model)
            logger.info("[GNNTransPlanner] 塌陷探针已挂载 -> %s", self._probe.path)

    @classmethod
    def get_requirements(cls) -> List[Type[Observation]]:
        return [DetectionsTracks]

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        # VT-2026-04191 D1-2：route 契约扩展。原实现丢弃 route_roadblock_ids（阶段⑨ 归因中
        # 「route/reference-line 缺失」列为闭环 drivable_area_compliance=0.296 的首要结构性缺口）。
        self._route_roadblock_ids = list(initialization.route_roadblock_ids or [])
        logger.info("[GNNTransPlanner] route_roadblock_ids=%d 条", len(self._route_roadblock_ids))
        self._load_model()

    def name(self) -> str:
        return "GNNTransPlanner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    # ---- 主入口：模型推理 → 参考轨迹（无异常吞噬）----
    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        self._load_model()
        from torch_geometric.data import Batch

        graph_data = self._build_graph_data(current_input)
        batch = Batch.from_data_list([graph_data]).to(self._device)  # data.batch=zeros(N)，batch_size=1
        with torch.no_grad():
            output = self._model(batch)  # [1, 120] = 60×2 局部增量
        # Q46：落盘发生在前向**之后**、轨迹构建之前；hook 返回 None ⇒ output 未被改动
        if self._probe is not None:
            self._probe.record(self._scenario_id, int(current_input.iteration.index))
        return self._output_to_trajectory(output, current_input)

    # ---- 在线特征构建：完全对齐 E0 训练契约 + 复用训练同一图构建 helper ----
    def _build_graph_data(self, current_input: PlannerInput):
        config = _load_gnn_config()
        from src.shared.utils.data_utils import (
            sampled_past_ego_states_to_tensor,
            sampled_past_timestamps_to_tensor,
            sampled_tracked_objects_to_tensor_list,
            agent_past_process,
            get_neighbor_vector_set_map,
            map_process,
            encoding_vectornet_features,
            get_route_lane_progress,
            build_route_channels,
            reorder_lane_ids,
        )
        from src.shared.utils.dataset import build_graph_data_from_feature_df

        history = current_input.history
        current_ego_state = history.ego_states[-1]
        if not self._diag_logged:  # 首帧核对 history 长度是否满足 20 帧契约
            logger.info("[GNNTransPlanner] history 首帧: ego_states=%d observations=%d",
                        len(history.ego_states), len(history.observations))
            self._diag_logged = True

        # 1. ego 历史 20帧/2s（E0 契约；模型 ego 分支锁 20 帧）
        ego_states = list(history.ego_states[-config.NUM_PAST_POSES:])
        past_ego_states = sampled_past_ego_states_to_tensor(ego_states)
        past_timestamps = sampled_past_timestamps_to_tensor(ego_states)

        # 2. 历史 tracked objects：传 DetectionsTracks 本身（与预处理 get_tracked_objects_at_iteration 同类型；
        #    _extract_agent_tensor 会取其 .tracked_objects=TrackedObjects。切勿传 obs.tracked_objects，否则再钻一层进普通 list）
        past_tracked_objects = list(history.observations[-config.NUM_PAST_POSES:])
        while len(past_tracked_objects) < config.NUM_PAST_POSES:
            past_tracked_objects.insert(0, DetectionsTracks(TrackedObjects([])))  # 空帧占位
        tracked_list, tracked_types = sampled_tracked_objects_to_tensor_list(
            past_tracked_objects[-config.NUM_PAST_POSES:]
        )

        # 3. agents=20（E0 契约，原 planner 用 15）
        ego_past, agents_past, _ = agent_past_process(
            past_ego_states, past_timestamps, tracked_list, tracked_types, config.NUM_AGENTS
        )

        # 4. 地图 LANE-only / radius=50 / max LANE 40×50（E0 契约；tl 仍传 [] 与训练同）
        #    route 契约：传入真实 route_roadblock_ids 并取回与 coords['LANE'] 同序的 lane_ids，
        #    据此生成 route 三通道（方案B：写 lane 节点 col6/col7/col8，不扩节点维度）。
        ego_position = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)
        coords, tl_data, lane_ids = get_neighbor_vector_set_map(
            self._map_api, config.MAP_FEATURES, ego_position, config.MAP_RADIUS,
            self._route_roadblock_ids, [], return_lane_ids=True,
        )
        route_lane_progress = get_route_lane_progress(
            self._map_api, ego_position, config.MAP_RADIUS, self._route_roadblock_ids
        )
        # 取回 selected_indices —— 生效的 convert_feature_layer_to_fixed_size 会按到 ego 的距离
        # 重排并截断 lane，route 三通道必须用同一 permutation 对齐（2026-07-28 事故修复）。
        vector_map, selected_indices = map_process(
            current_ego_state.rear_axle, coords, tl_data, config.MAP_FEATURES,
            config.MAX_ELEMENTS, config.MAX_POINTS, config.INTERPOLATION,
            return_selected_indices=True,
        )
        if self._route_mode == "three_channel":
            ordered_lane_ids = reorder_lane_ids(lane_ids, selected_indices.get("LANE", []))
            lane_route_channels = build_route_channels(
                ordered_lane_ids, vector_map["lanes"],
                route_lane_progress, config.LANE_NUM, config.MAP_RADIUS,
            )
        else:  # route_mode="none"：强制全零，复现无 route 的旧基线
            lane_route_channels = None

        # 5. EGO_HISTORY [20,2] 局部逐帧增量（决策D；复刻预处理 current_heading 参考系）
        ego_history_local = self._build_ego_history_local(
            ego_states, current_ego_state.rear_axle.heading
        )
        ego_future_placeholder = np.zeros((config.NUM_FUTURE_POSES, 2), dtype=np.float32)  # 推理不需 GT
        t_ref = past_timestamps[-1].item()

        feature_df = encoding_vectornet_features(
            agents_past, None, ego_past, ego_future_placeholder, vector_map,
            past_timestamps, t_ref, ego_history=ego_history_local,
            lane_route_channels=lane_route_channels,
        )

        # 6. 与训练同一图构建（gnn_x 均值池化 + DefaultBuilder 无向补反向 + cluster col9）
        graph_data, skip_reason = build_graph_data_from_feature_df(feature_df, self._adjacency_config)
        if skip_reason is not None:
            raise RuntimeError(
                f"[GNNTransPlanner] 在线图构建被跳过（{skip_reason}）：无有效 polyline，"
                f"ego@({ego_position.x:.1f},{ego_position.y:.1f})——失败即失败，不静默替换。"
            )
        return graph_data

    @staticmethod
    def _build_ego_history_local(ego_states: list, current_heading: float) -> torch.Tensor:
        """[20,2] 车辆局部逐帧增量，参考朝向=current_heading（预处理 L119-127 同约定）。

        全局→局部：local_dx = g_dx·cos_h + g_dy·sin_h；local_dy = -g_dx·sin_h + g_dy·cos_h。首帧 [0,0]。
        仿真早期 history 不足 20 帧时前补 [0,0]（模型 ego 分支须 [20,2]）。
        """
        cos_h, sin_h = np.cos(current_heading), np.sin(current_heading)
        inc = [[0.0, 0.0]]
        for i in range(1, len(ego_states)):
            g_dx = ego_states[i].rear_axle.x - ego_states[i - 1].rear_axle.x
            g_dy = ego_states[i].rear_axle.y - ego_states[i - 1].rear_axle.y
            inc.append([g_dx * cos_h + g_dy * sin_h, -g_dx * sin_h + g_dy * cos_h])
        arr = np.asarray(inc, dtype=np.float32)
        n_pad = 20 - arr.shape[0]
        if n_pad > 0:
            arr = np.concatenate([np.zeros((n_pad, 2), dtype=np.float32), arr], axis=0)
        return torch.from_numpy(arr[-20:])

    # ---- C4 规划层接线（设计文档 §8 第 2 步）----
    def _refine_reference(self, ref_xy: np.ndarray, current_input: PlannerInput,
                          current_state: EgoState):
        """把上游参考轨迹交给 C4 规划层精化，返回 `(gx, gy)`。

        三个上游输入分别来自：
          * agent 占据 ← `current_input.history.observations[-1]`（`DetectionsTracks` 常速前推）
          * 限速 / 可行驶区域 ← `self._map_api`
          * route 中心线 ← `self._route_roadblock_ids`（**上游塌陷时的兜底几何**）

        🔴 本方法只在规划层开启或上游切成 CV 时才会被调用；默认路径一行都不执行。
        """
        from src.planning_layer import constant_velocity_reference, refine
        from src.planning_layer import nuplan_adapter as adapter

        ego = adapter.ego_snapshot(current_state)
        cfg = self._planning_layer_cfg
        if self._reference_source == "cv":
            # CV 上游：旁路模型输出，只用当前状态常速外推（层可再单独开关，故与 cfg.enabled 无关）
            ref_xy = constant_velocity_reference(ego, cfg)
        if not cfg.enabled:
            return ref_xy[:, 0], ref_xy[:, 1]

        if self._drivable is None:
            self._drivable = adapter.DrivableAreaClosure(self._map_api)
            self._route_centerline = adapter.RouteCenterline(self._map_api)
        ctx = adapter.build_scene_context(
            current_state, current_input.history.observations[-1], self._map_api,
            horizon=cfg.horizon, dt=cfg.dt, drivable=self._drivable,
            speed_limit=adapter.speed_limit_mps(self._map_api, current_state),
        )
        route_xy = self._route_centerline.get(self._route_roadblock_ids, (ego.x, ego.y))
        out = refine(ref_xy, ego, ctx=ctx, cfg=cfg, route_xy=route_xy)
        return out[:, 0], out[:, 1]

    # ---- 模型增量 → 全局参考轨迹（不搓运动学/不夹速度；可行性交 two_stage_controller）----
    def _output_to_trajectory(
        self, model_output: torch.Tensor, current_input: PlannerInput
    ) -> InterpolatedTrajectory:
        current_state = current_input.history.ego_states[-1]
        inc = model_output.detach().cpu().numpy().reshape(-1, 2)  # [60,2] 局部逐帧增量（首帧≈[0,0]）
        local_pos = np.cumsum(inc, axis=0)                        # [60,2] 局部累计位移（X前/Y左）

        h = current_state.rear_axle.heading
        cos_h, sin_h = np.cos(h), np.sin(h)
        ox, oy = current_state.rear_axle.x, current_state.rear_axle.y
        # 局部→全局（预处理「全局→局部」的逆变换）
        gx = local_pos[:, 0] * cos_h - local_pos[:, 1] * sin_h + ox
        gy = local_pos[:, 0] * sin_h + local_pos[:, 1] * cos_h + oy

        # ---- C4 规划层（R4-1）：默认**完全不执行**，故关闭时输出逐位不变（判据 [L]）----
        if self._planning_layer_cfg.enabled or self._reference_source != "model":
            gx, gy = self._refine_reference(np.stack([gx, gy], axis=1), current_input, current_state)

        # 训练原点=future[0]=t+0.1（increments[0]=[0,0]），故 cumsum[i] 是 t+0.1·(i+1) 相对 future[0] 的位置。
        # 在线锚定当前 ego：把 cumsum[i] 放在 t0+i·dt（cumsum[0]≈当前 ego），则首个未来点=cumsum[1]=真实约 0.9m 位移，
        # 消除「首点=[0,0]→控制器每步零速→ego 闭环停滞」滞后（cumsum[0] 放 t0+0.1 会令 progress~5%）。
        trajectory: List[EgoState] = [current_state]  # t0（≈ cumsum[0]）
        t0 = current_state.time_point.time_us
        dt_us = int(self._sampling_time * 1e6)
        veh_params = current_state.car_footprint.vehicle_parameters
        prev_x, prev_y, prev_h = ox, oy, h
        for i in range(1, self._num_output_points):  # cumsum[1..59] → t0+0.1 .. t0+5.9
            x, y = float(gx[i]), float(gy[i])
            dx, dy = x - prev_x, y - prev_y
            dist = float(np.hypot(dx, dy))
            heading = float(np.arctan2(dy, dx)) if dist > _HEADING_EPS else prev_h
            speed = dist / self._sampling_time  # 车体纵向速度 ≈ |Δ|/dt
            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(speed, 0.0),  # 车体系：纵向 speed，横向 0
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + i * dt_us),
                vehicle_parameters=veh_params,
                is_in_auto_mode=True,
                angular_vel=0.0,
                angular_accel=0.0,
            )
            trajectory.append(state)
            prev_x, prev_y, prev_h = x, y, heading
        return InterpolatedTrajectory(trajectory)
