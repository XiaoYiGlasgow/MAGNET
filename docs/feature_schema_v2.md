# Feature Schema v2 — Freeze Record

> Freeze date: 2026-07-28 | Belongs to: VT-2026-04191 Stage 15 D0#4 / Z7
> **schema hash: `31bce53e29f9f0bd`**

---

## 0. Single source of truth

This document is **not** the definition of the schema; it is only a freeze record. The definition lives in the code:

```
scripts/analysis/result_manifest.py :: FEATURE_SCHEMA_V2
```

If that dictionary is changed, the hash changes, and every result JSON that references the old hash will be
automatically flagged in `validate_manifest()` as "input contract has changed; cannot be merged into the same table directly".

Inspecting and retrieving values:

```bash
GNN=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans; PY=/home/team/anaconda3/envs/nuplan/bin/python
$PY $GNN/scripts/analysis/result_manifest.py --print_schema        # human-readable list
$PY $GNN/scripts/analysis/result_manifest.py --print_schema_hash   # 31bce53e29f9f0bd
$PY $GNN/scripts/analysis/result_manifest.py --self_test           # self-test
```

---

## 1. Why a hash is needed

The semantics of `col6/7/8` **branch by node type**:

- On **ego / agent** rows they are `timestamp / v_start / v_end` (this has been the case since v1 and is unchanged);
- On **lane** rows, v1 has three placeholder slots that are always 0, and **v2 changes them to carry the three route channels**.

This means that whether a `features_<token>.pkl` is v1 or v2, and which semantics a checkpoint saw during training,
**cannot be told from the data itself** — both have shape `(N, 12)` and dtype float32,
and with `route_mode=none` they are even byte-for-byte identical.

Therefore the normalised serialised hash of the schema must be written into every result JSON. Otherwise, six months later it will be impossible to answer
"which input contract does this table use", and that is exactly the question a reviewer will ask.

---

## 2. Column definitions (snapshot of `--print_schema`)

```
 col name                      ego/agent                              lane
 ---------------------------------------------------------------------------------------
 0   x_start                   segment start x                        segment start x
 1   y_start                   segment start y                        segment start y
 2   x_end                     segment end x                          segment end x
 3   y_end                     segment end y                          segment end y
 4   heading_start             start heading (rad)                    start heading (rad)
 5   heading_end               end heading (rad)                      end heading (rad)
 6   ts_or_route_proximity     normalised timestamp ∈ [0,1)           ★v2: route proximity clip(1-d/radius,0,1)
 7   vstart_or_route_progress  v_start (m/s)                          ★v2: normalised progress along route ∈ [0,1]
 8   vend_or_is_on_route       v_end (m/s)                            ★v2: is_on_route ∈ {0,1}
 9   subgraph_id               subgraph ID (cluster)                  subgraph ID (cluster)
 10  t_start                   start time relative to T_ref (s)       always 0 (placeholder)
 11  t_end                     end time relative to T_ref (s)         always 0 (placeholder)
```

- Coordinate frame: ego-centric, origin = current ego `rear_axle` pose, `x` = ego heading (forward), `y` = left.
- **Row semantics = one segment (two adjacent points) of a polyline, not one time instant.** This is often misread.
- Code reference: the three `np.hstack` calls in `src/shared/utils/data_utils.py::encoding_vectornet_features`
  (ego :826-837 / agent :869-878 / lane :904-915).

The two values of `route_mode`:

| route_mode | col6/7/8 on lane rows | Guarantee |
|---|---|---|
| `none` | **all 0** | **byte-for-byte identical** to the output before route was introduced (E0 criterion 0, re-verified 10/10) |
| `three_channel` | three route channels | ego/agent rows unchanged (E0 criteria 1-6, re-verified 10/10) |

Model side: `_extract_core_features` takes only the **first 9 columns**, so the three route channels take effect through the Transformer branch
**per sample** (`transformer_predictor.py:232-243` masks by `data.batch == i`).

---

## 3. Known limitations (must go into the paper's limitations; must not be omitted)

1. The three route channels **reuse the placeholder channels of lane nodes** and are **not an independent route branch**
   — the paper must not describe this as reference-line conditioning; if route is kept, Eq.(8) must be rewritten accordingly.
2. **About 11.8% of samples have an empty route signal** (95% CI [10.7%, 13.0%]),
   caused by the radius-connectivity pruning in devkit's `prune_route_by_connectivity`.
3. **About 19.9% of samples have more lanes than `LANE_NUM=40`**; these are sorted by distance to ego and truncated.
   The truncation itself does not affect the correctness of the route channels (the three channels are aligned with the same permutation), but it should go into the limitations.
4. `EGO_HISTORY` is `[20, 2]` local per-frame increments, **2 s @ 10 Hz** — the paper **must not state a 4 s history**.

---

## 4. Execution contract constants

| Item | Executed value |
|---|---|
| `NUM_AGENTS` | **20** |
| `LANE_NUM` × points | **40 × 50** |
| route lanes | **0 (do not exist)** |
| crosswalks | **0 (do not exist)** |
| `QUERY_RADIUS` | **50 m** |
| nearby-only threshold | **30 m** |
| `PAST_TIME_HORIZON` | 2 s |

The two constants `ROUTE_LANES_NUM=10` / `CROSSWALKS_NUM=5` exist in `config.py`, but they are
consumed only by the legacy `src/danger_generation/` line and are not used by MAGNET.

---

## 5. Freeze policy

- This schema was **frozen on 2026-07-28** for use in G3 final (4 cells × 5 seeds) and all subsequent formal experiments.
- **Any** change to `FEATURE_SCHEMA_V2` must:
  1. increment `SCHEMA_VERSION`;
  2. append a section to this file recording the v2→v3 delta and the new hash;
  3. re-run the E0 ten-sample acceptance check (`e0_verify.py` + `e0_route_diff.py` + `e0_infer.py`;
     the four-topology edge hashes must remain `861/41/40/377`);
  4. **never** merge results with different schema hashes into the same table.
- Every formal result JSON is produced by `build_result_manifest()` and must contain a `schema_hash` field;
  `validate_manifest()` raises an error when the hash drifts.
