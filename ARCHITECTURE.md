# Social-LOVON Architecture

## Overview

Social-LOVON is a socially-aware robot navigation system. A **language-to-motion transformer** (L2MM) converts natural-language mission instructions into velocity commands `[vx, vy, ωz]`. A **SocialNavigator** then intercepts those commands and modulates the angular rate to steer the robot along a trajectory that avoids detected humans.

Everything is orchestrated by `deploy.py`. The `--crowdnav_sim_mode` flag swaps the full real sensor stack for a CrowdNav gym simulation, letting the complete pipeline (including SocialNavigator) run without hardware.

---

## deploy.py Pipelines

```
┌──────────────────────────────────┐        ┌─────────────────────────────────────────┐
│         REAL ROBOT MODE          │        │          CROWDNAV SIM MODE              │
│        (default)                 │        │        (--crowdnav_sim_mode)            │
│                                  │        │                                         │
│  ImageGetterThread               │        │  CrowdNavDataProvider.step(mv)          │
│    └─ robot camera / webcam      │        │    ├─ env.step()  (ORCA human agents)   │
│  YoloProcessingThread            │        │    ├─ _generate_synthetic_pose_state()  │
│    └─ YOLO11 object detection    │        │    ├─ _generate_synthetic_lidar()       │
│  YoloPoseProcessingThread        │        │    └─ _generate_synthetic_object_state()│
│    └─ YOLO pose keypoints        │        │                                         │
│  LiDARGetterThread               │        │  Synthetic pose_state + lidar + YOLO   │
│    └─ UDP PointCloud2 (10-scan   │        │  fed directly into _update_motion_      │
│         accumulate buffer)       │        │  control() — no sensor threads          │
└──────────┬───────────────────────┘        └──────────────────┬──────────────────────┘
           │                                                   │
           │  result_queue.get(state)   OR   _crowdnav_tick()  │
           └───────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                    _update_motion_control(state, lidar_cloud)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             MotionPredictor              SocialNavigator
               .predict(data)               .step(mv, pose_state,
                    │                            mission_state, lidar)
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                              motion_vector [vx, vy, ωz]
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │ REAL: sport_client      │
                      │       .Move(vx, vy, ωz) │
                      │ SIM:  stored for next    │
                      │       env.step()         │
                      └────────────────────────┘
```

**Key threads (real mode):**

| Thread | Role |
|---|---|
| `ImageGetterThread` | Pulls frames from robot camera / webcam; Laplacian blur filter; feeds `image_queue` |
| `YoloProcessingThread` | YOLO11 object detection → `state` dict (bbox, xyn, confidence) |
| `YoloPoseProcessingThread` | YOLO pose → `pose_state` dict (keypoints, boxes, num_people) |
| `LiDARGetterThread` | UDP PointCloud2 subscriber; accumulates last 10 scans; exposes `get_cloud()` |
| `MotionControlThread` | Consumes `result_queue`; calls `_update_motion_control()` + `_control_robot()` |

---

## SocialNavigator — Full Architecture

`models/api_social_navigator.py`

### Classes

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  TrackedHuman                                    data class, one instance per track  │
│                                                                                      │
│  track_id        int                             persistent ByteTrack ID             │
│  position_image  (cx, cy) px                     bbox centroid in image              │
│  bbox            (x1,y1,x2,y2) px                                                   │
│  keypoints       (17,2) COCO keypoints                                               │
│  keypoints_conf  (17,) per-keypoint confidence                                       │
│  confidence      float                           YOLO detection confidence           │
│  distance_lidar  float | None  m                 from LiDAR cluster estimate         │
│  distance_mono   float | None  m                 from mono: mono_k / bbox_h          │
│  distance        float | None  m                 fused (lidar preferred)             │
│  position_rf     [x_lat, depth] m                robot-frame 2D position            │
│  velocity        [vx, vy] m/s                    estimated from history delta        │
│  predicted_path  [[x,y], ...]                    future positions from predictor     │
│  orientation     float rad                       body heading (unused)               │
│  last_seen       float timestamp                                                     │
│  is_ghost        bool                            True = predicted, not observed       │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  SocialNavigator                                 main socially-aware nav layer       │
│                                                                                      │
│  __init__(enabled, **kwargs)                                                         │
│    ├─ camera intrinsics  fx, fy, cx, cy  from fov_deg + image_width/height          │
│    ├─ ByteTrack state    _byte_tracks, _tracked_humans, _next_id                    │
│    ├─ HumanTrajectoryPredictor  _predictor                                          │
│    ├─ Shield state       shield_active=False, safety_score=1.0                      │
│    ├─ Lidar cache        _lidar_image_points, _lidar_human_masks                    │
│    ├─ Trajectory state   _current_traj, _best_traj, _goal_rf, _ego_velocity         │
│    └─ Diagnostics dict   diag{num_humans, min_distance, shield_active, ...}         │
│                                                                                      │
│  step(motion_vector, pose_state, mission_state, lidar_ranges) → motion_vector       │
│    └─ 8-stage pipeline (see below)                                                  │
│                                                                                      │
│  update_goal(object_xyn, bbox_height_px)                                            │
│    └─ sets _goal_rf = [x_lat, depth] from mono_k / bbox_h + pinhole                │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

### `step()` — 8-Stage Pipeline

```
step(motion_vector, pose_state, mission_state, lidar_ranges)
│
│  [always runs — so BEV shows humans even when disabled]
│
├─ Stage 1: Parse detections
│    _parse_pose_state(pose_state)
│      ├─ iterate pose_state["poses"] + ["pose_boxes"]
│      └─ build det dict: {bbox, center_px, bbox_height, keypoints, keypoints_conf, confidence}
│
├─ Stage 2: Distance estimation
│    _project_lidar_to_image(lidar_ranges)
│      ├─ filter: finite, non-origin, lx>0 (front-facing), z_min < lz < z_max
│      ├─ apply yaw + pitch calibration offsets (lidar_cam_yaw_offset, pitch_offset)
│      ├─ pinhole projection → (u_norm, v_norm) in [0,1]
│      └─ returns (N,5) array: [u_norm, v_norm, lx, ly, lz]  → cached as _lidar_image_points
│
│    _estimate_distances(detections, lidar_ranges)
│      for each detection:
│        ├─ _estimate_distance_lidar(det, lidar_ranges)
│        │    ├─ skeleton-matching: min distance from lidar points to COCO skeleton segments
│        │    │    (uses lidar_kpt_conf_thresh, lidar_skeleton_dist)
│        │    ├─ fallback: normalized bbox inclusion
│        │    ├─ nearest-cluster: percentile seed + cluster_margin to reject walls
│        │    └─ return median depth of cluster  → mask saved in _lidar_human_masks
│        ├─ _estimate_distance_mono(det)
│        │    └─ d = mono_k / bbox_height_px
│        ├─ fuse: lidar preferred, mono fallback
│        └─ _pixel_to_robot_frame(det)
│               x_lat = depth * (u - cx) / fx
│               position_rf = [x_lat, depth]
│
├─ Stage 3: ByteTrack tracker
│    _update_tracker(detections)
│      ├─ split: high_conf (≥ track_high_thresh) / low_conf (≥ track_low_thresh)
│      ├─ split tracks: active / lost
│      │
│      ├─ Pass 1: high_conf dets vs active tracks  (IoU ≥ track_iou_thresh)
│      │    _associate() → greedy Hungarian on IoU matrix (_compute_iou_matrix)
│      │    _apply_detection(): copy det fields; EMA smooth lidar distance
│      │
│      ├─ Pass 2: low_conf dets vs remaining active tracks
│      │    unmatched active → state="lost", frames_lost++
│      │
│      ├─ Pass 3: remaining high_conf dets vs lost tracks  (re-ID)
│      │    matches → state="active", frames_lost=0
│      │
│      ├─ Unmatched high_conf dets → new tracks (_next_id++)
│      ├─ Unmatched lost tracks → frames_lost++ ; prune if > track_max_lost
│      └─ Rebuild _tracked_humans{id: TrackedHuman} from active _byte_tracks
│
├─ Stage 4: Trajectory prediction
│    _predict_trajectories()
│      ├─ _compensate_ego_motion()
│      │    translate + rotate predictor history from prev robot frame to current:
│      │      x_lat += v_lat*dt;  depth -= v_fwd*dt
│      │      then rotate by -omega*dt (cos/sin rotation)
│      │
│      ├─ feed each tracked human's position_rf into _predictor.update_agent_position()
│      ├─ _predictor.predict_all() → linear extrapolation (polyfit degree 1)
│      ├─ store predicted_path, velocity (delta / dt) in each TrackedHuman
│      │
│      └─ Ghost humans (out-of-FOV persistence):
│           for each predictor ID not in current active tracks:
│             ├─ prune if frames_since_seen > ghost_max_frames (default 400)
│             ├─ index into predicted path at frames_since_seen - 1
│             ├─ de-duplicate: if ghost_pos within 1m of an observed human, merge
│             └─ create ghost TrackedHuman(is_ghost=True) at predicted position
│
│  [stages 5–8 only run when enabled=True]
│
├─ Stage 5: Safety score
│    _compute_safety_score()
│      ├─ collect human_positions (position_rf) and human_predicted_paths
│      └─ robot_safety_score(0, 0, positions, paths)
│           → calls safety.py: Gaussian proximity + trajectory threat field at (0,0)
│           → returns float in [0, 1]  (1 = fully safe, 0 = imminent collision)
│
│    get_safety_heatmap()
│      └─ compute_safety_grid() → 2D grid stored as self.grid  (used by BEV + traj)
│
├─ Stage 6: Shield gate
│    _evaluate_shield(mission_state)
│      ├─ only arms when mission_state in shield_active_states (default: ["running"])
│      └─ hysteresis:
│           activate:   safety_score < shield_thresh_on   (default 1.0)
│           deactivate: safety_score > shield_thresh_off  (default 1.5)
│
├─ Stage 6.5: Trajectory data
│    _update_trajectory_data()
│      ├─ _extrapolate_robot_path_full(motion_vector)
│      │    → cubic Bezier from (0,0) to _goal_rf, tangent to forward heading
│      │    → tuned by path_curvature param
│      │    score via _trajectory_eval()  → arc-length-weighted average safety
│      │
│      └─ _get_best_traj(traj_type="elastic")
│             _construct_trajectory()
│               ├─ Phase 1 — gradient walk
│               │    start at (0,0) in departure direction (from ego velocity)
│               │    each step: heading + grad_gain*∇safety + goal_gain*(goal−cur)
│               │    stop when safety ≥ shield_thresh_off OR past goal distance
│               ├─ _clamp_curvature(): hard-limit Menger curvature via binary search
│               ├─ Phase 2 — cubic Bezier from walk endpoint to goal
│               └─ returns blended path: gradient-walk + Bezier tail
│             _trajectory_eval_v2()
│               → vectorized _safety_scores() over all points
│               → arc-length-weighted average safety score
│
├─ Stage 7: Command correction
│    _correct_command(motion_vector)   [only when shield_active]
│      └─ _omega_from_trajectory(motion_vector)
│              sample 3 evenly-spaced points from _best_traj
│              signed Menger curvature: κ = 2·cross(AB, BC) / (|AB|·|BC|·|AC|)
│              omega = v_fwd * κ     (clip to ±max_omega_mag)
│         returns [vx_unchanged, vy_unchanged, omega_corrected]
│
└─ Stage 8: Diagnostics
     _update_diagnostics()
       → populates self.diag: num_humans, min_distance, safety_score,
                              shield_active, traj_score, best_traj_score
```

---

### Visualization Methods

```
render_bev(show_heatmap=True) → 400×400 BGR image
  ├─ Safety heatmap underlay (RdYlGn colormap from self.grid)
  ├─ shield_thresh_on  contour (black)
  ├─ shield_thresh_off contour (black)
  ├─ Camera FOV lines (grey)
  ├─ Range rings at 1m intervals (grey)
  ├─ LiDAR point cloud (JET colormap by Z height)
  ├─ LiDAR Z-height colorbar legend
  ├─ Robot marker  (green triangle)
  ├─ Original motion vector path (yellow)
  ├─ Best trajectory / elastic-band (cyan)
  ├─ Corrected motion path (cyan)
  ├─ Observed humans (red filled circles + ID + distance)
  ├─ Ghost humans    (blue hollow circles + "(ghost)" label)
  ├─ Predicted human trajectories (orange dots)
  └─ Goal marker (green star)

overlay_lidar(image) → annotated camera image
  ├─ reads _lidar_image_points (shared projection from step())
  ├─ non-human points: coloured circles (JET by forward distance)
  └─ human-matched points: white star markers

get_safety_heatmap(xlim, ylim, resolution) → (grid, extent)
  └─ calls compute_safety_grid() from safety.py; stores result as self.grid
```

---

### Tunable Parameters (`DEFAULT_PARAMS`)

| Category | Key | Default | Description |
|---|---|---|---|
| Shield | `shield_thresh_on` | 1.0 | Safety score below → shield activates |
| Shield | `shield_thresh_off` | 1.5 | Safety score above → shield deactivates |
| Shield | `shield_active_states` | `["running"]` | Mission states where shield is armed |
| Correction | `correction_gain` | 25.0 | Potential-field omega gain (legacy) |
| Correction | `bezier_omega_gain` | 1.0 | Bezier-curvature omega gain |
| Correction | `max_omega_mag` | 1.0 | Omega clamp (rad/s) |
| Robot path | `horizon_s` | 2.0 | Unicycle prediction horizon (s) |
| Robot path | `horizon_steps` | 10 | Unicycle prediction steps |
| Robot path | `path_curvature` | 0.45 | Bezier tangent handle scale |
| Camera | `image_width` | 640 | px |
| Camera | `image_height` | 480 | px |
| Camera | `fov_deg` | 80.0 | Horizontal FOV |
| Camera | `fov_v_deg` | 45.0 | Vertical FOV |
| Predictor | `pred_history` | 25 | Position history frames |
| Predictor | `pred_steps` | 60 | Future steps to predict |
| ByteTrack | `track_high_thresh` | 0.5 | High-conf association threshold |
| ByteTrack | `track_low_thresh` | 0.1 | Low-conf association threshold |
| ByteTrack | `track_iou_thresh` | 0.3 | Min IoU to accept match |
| ByteTrack | `track_max_lost` | 30 | Frames before track pruned |
| LiDAR | `use_lidar_depth` | True | Use LiDAR for depth (vs mono only) |
| LiDAR | `lidar_z_min/max` | -0.3 / 5.0 | Z filter bounds (m) |
| LiDAR | `lidar_ema_alpha` | 0.5 | Distance EMA smoothing |
| LiDAR | `lidar_cluster_margin` | 0.5 | Wall-rejection cluster margin (m) |
| LiDAR calib | `lidar_cam_yaw_offset` | 0.0 | Yaw calibration offset (deg) |
| LiDAR calib | `lidar_cam_pitch_offset` | 1.0 | Pitch calibration offset (deg) |
| LiDAR calib | `lidar_cam_z_offset` | 0.05 | Camera height above LiDAR (m) |
| BEV | `bev_range_m` | 7.0 | Forward BEV range (m) |
| BEV | `bev_behind_m` | 2.0 | Behind BEV range (m) |
| Ego-motion | `time_step` | 0.25 | Control cycle duration (s) |
| Ghost | `ghost_max_frames` | 400 | Max frames a ghost persists |
| Trajectory | `traj_step_size` | 0.2 | Gradient-walk step size (m) |
| Trajectory | `traj_gradient_gain` | 0.5 | Safety-gradient influence |
| Trajectory | `traj_goal_gain` | 0.3 | Goal attraction strength |
| Trajectory | `traj_max_steps` | 100 | Max gradient-walk steps |
| Trajectory | `max_traj_curvature` | 1.0 | Menger curvature limit |
| Mono depth | `mono_k` | 300.0 | Monocular depth constant (px·m) |

---

### Key Data Flow (per control cycle)

```
pose_state  ──┐
lidar_cloud ──┤
               ▼
          _parse_pose_state()
               │ detections: [{bbox, kpts, conf, ...}, ...]
               ▼
          _project_lidar_to_image()
               │ _lidar_image_points (N,5)
               ▼
          _estimate_distances()
               │ + distance_lidar (skeleton/bbox cluster)
               │ + distance_mono  (mono_k / bbox_h)
               │ + distance       (fused, EMA smoothed in tracker)
               │ + position_rf    [x_lat, depth]
               ▼
          _update_tracker()  [ByteTrack 3-pass IoU]
               │ _tracked_humans {id: TrackedHuman}
               ▼
          _compensate_ego_motion()
               │ all predictor history rotated to current robot frame
               ▼
          HumanTrajectoryPredictor.predict_all()
               │ predicted_path for each human
               ▼
          ghost logic
               │ out-of-FOV agents persist via trajectory extrapolation
               ▼
          [if enabled]
               │
          _compute_safety_score()          ─┐
               │  safety_score [0,1]         │  safety.py:
          get_safety_heatmap()              │    robot_safety_score()
               │  self.grid                  │    compute_safety_grid()
               ▼                             │    Gaussian proximity +
          _evaluate_shield()                │    trajectory threat
               │  shield_active bool        ─┘
               ▼
          _update_trajectory_data()
               │  _current_traj   (Bezier to goal, scored)
               │  _best_traj      (elastic-band: gradient walk + Bezier tail)
               ▼
          _correct_command()    [only if shield_active]
               │  _omega_from_trajectory():
               │    Menger curvature of _best_traj → omega = v_fwd * κ
               ▼
          return [vx, vy, omega_corrected]
```

---

### Ghost Humans

When a tracked human exits the camera FOV, their predictor history is retained. On subsequent frames the ghost's "current" position is read by indexing into the predicted trajectory at `frames_since_seen - 1`. A ghost `TrackedHuman` (`is_ghost=True`) is created at that position and participates in the safety grid and BEV visualization (hollow blue circle). Ghosts expire after `ghost_max_frames` (default 400) or when the predicted position falls outside the prediction horizon. De-duplication: if a ghost's predicted position is within 1 m of a newly observed human, the stale predictor entry is pruned.

### Shield Hysteresis

Two thresholds prevent oscillation:
- **Activate** when `safety_score < shield_thresh_on` (default **1.0**)
- **Deactivate** when `safety_score > shield_thresh_off` (default **1.5**)

Shield only arms when `mission_state in shield_active_states` (default `["running"]`).

### Trajectory Correction

When the shield is active, `_correct_command()` replaces the L2MM omega with one derived from the best elastic-band trajectory:
1. `_construct_trajectory()` walks forward on the safety heatmap gradient, then splices a cubic Bezier to the goal.
2. `_trajectory_eval_v2()` scores the trajectory as the arc-length-weighted average safety (vectorized).
3. `_omega_from_trajectory()` samples 3 equidistant points near the start of `_best_traj`, computes signed Menger curvature κ, and returns `omega = v_fwd * κ`.

---

## Module Map

```
Social-LOVON/
├── deploy/
│   └── deploy.py                          # Entry point; VisualLanguageController + all threads
├── models/
│   ├── api_social_navigator.py            # SocialNavigator + TrackedHuman
│   ├── api_language2mostion.py            # L2MM MotionPredictor (transformer → [vx,vy,ωz])
│   ├── api_object_extraction.py           # SequenceToSequenceClassAPI (text → "handbag")
│   ├── humantrajectorypredictor.py        # HumanTrajectoryPredictor (polyfit-1 linear)
│   ├── safety.py                          # Gaussian + trajectory safety field functions
│   └── crowdnav_data_provider.py          # Synthetic perception from CrowdNav (sim mode)
├── configs/
│   ├── env_lovon.config                   # CrowdSim environment parameters
│   └── policy_lovon.config                # Policy + kinematics config
├── crowd_sim/envs/                        # CrowdNav gym environment
│   ├── crowd_sim.py                       #   CrowdSim(gym.Env)
│   └── utils/                             #   Agent, Robot, Human, State, Action
└── crowd_nav/                             # CrowdNav policy registry + Explorer
```

---

## Supporting Modules

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  HumanTrajectoryPredictor                                                           │
│  (models/humantrajectorypredictor.py)                                               │
│                                                                                     │
│  update_agent_position(id, [x,y], timestep)  append to sliding deque window        │
│  predict_trajectory(id) → [[x,y], ...]       polyfit(deg=1) linear extrapolation   │
│  predict_all(timestep)  → {id: [[x,y],...]}  predict all agents (throttled)        │
│  prune_stale(active_ids)                      remove gone agents                   │
│  reset()                                      clear all state                      │
│                                                                                     │
│  State: agent_trajectories = {id: deque([{position, timestep}, ...])}              │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  safety.py  (models/safety.py)                                                      │
│                                                                                     │
│  robot_safety_score(rx, ry, positions, paths) → float [0,1]                        │
│  safety_score_at_point(x, y, positions, paths) → float                             │
│  compute_safety_grid(positions, xlim, ylim, res) → (grid, extent)                  │
│  _gaussian_grid(X, Y, positions)    proximity: exp(-d²/2σ²),  σ=1.5m              │
│  _trajectory_grid(X, Y, paths)      predicted-path threat,  γ=0.995 decay          │
│  _safety_scores(X, Y, positions, paths)  min(gaussian, trajectory) → [0,1]         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  CrowdNavDataProvider  (models/crowdnav_data_provider.py)                           │
│  Only active with --crowdnav_sim_mode; replaces camera + LiDAR + YOLO              │
│                                                                                     │
│  __init__(env_config, policy_config, ...)  CrowdSim env with ORCA human agents     │
│  reset(phase, test_case) → obs             reset gym env                            │
│  step(motion_vector) → {pose_state, lidar, object_state, obs, reward, ...}         │
│  _generate_synthetic_pose_state(humans_rf)  sim positions → fake YOLO detections   │
│  _generate_synthetic_lidar(humans_rf)       sim positions → fake lidar cloud       │
│  _generate_synthetic_object_state(self_st)  goal → fake object detection           │
│  render_frame()                             matplotlib top-down view               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
