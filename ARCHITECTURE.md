# Social-LOVON Architecture

## Overview

Social-LOVON is a socially-aware robot navigation system built around two models: a **language-to-motion transformer** (L2MM) that converts natural-language instructions into velocity commands, and a **social navigator** that modulates those commands to avoid humans.

Everything runs through `deploy.py`. The `--crowdnav_sim_mode` flag swaps the real sensor stack for a CrowdNav gym simulation that generates synthetic YOLO/LiDAR data, exercising the full perception pipeline without hardware.

---

## Pipeline Comparison

```
                    REAL ROBOT                          CROWDNAV SIM MODE
                    (default)                           (--crowdnav_sim_mode)
              ──────────────────                    ──────────────────────────

              Go2 camera / RealSense                CrowdNavDataProvider
              LiDARGetterThread (UDP)                  .step(motion_vector)
              YoloProcessingThread                     ├─ env.step() (ORCA humans)
              YoloPoseProcessingThread                 ├─ _generate_synthetic_pose_state()
                     │                                 ├─ _generate_synthetic_lidar()
                     │                                 └─ _generate_synthetic_object_state()
                     ▼                                          │
              result_queue.get(state)                           │
                     │                                          │
                     ├─ state has real YOLO                     ├─ state has synthetic YOLO
                     │  detections + pose                       │  detections + pose
                     │                                          │
                     └──────────────┬───────────────────────────┘
                                    │
                                    ▼
                       _update_motion_control(state, lidar_cloud)
                                    │
                       ┌────────────┴─────────────┐
                       │                          │
                       ▼                          ▼
                 MotionPredictor            SocialNavigator
                   .predict()                  .step()
                       │                     (full 8-stage
                       │                      pipeline)
                       │                          │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                           motion_vector [vx, vy, wz]
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │ REAL: sport_client     │
                        │       .Move(vx,vy,wz)  │
                        │ SIM:  stored for next   │
                        │       env.step()        │
                        └───────────────────────┘
```

Key difference: in `--crowdnav_sim_mode`, the 4 sensor/detection threads are replaced by a single `_crowdnav_tick()` call inside MotionControlThread that steps the CrowdNav gym env and builds synthetic `pose_state`, `lidar`, and `object_state` dicts. The downstream pipeline (`SocialNavigator.step()` with all 8 stages) is identical in both modes.

---

## Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  VisualLanguageController                                                       │
│  (deploy/deploy.py)                                                             │
│                                                                                 │
│  __init__(model_dir, ..., crowdnav_sim_mode)                                    │
│    ├─ MotionPredictor (L2MM)                                                    │
│    ├─ SocialNavigator                                                           │
│    ├─ YOLO models (object + pose)           — skipped in sim mode               │
│    ├─ Unitree SDK (sport_client)            — skipped in sim mode               │
│    └─ CrowdNavDataProvider                  — only in sim mode                  │
│                                                                                 │
│  Threads:                                                                       │
│    ImageGetterThread                        — camera capture → image_queue       │
│    YoloProcessingThread                     — object detection (skipped sim)     │
│    YoloPoseProcessingThread                 — pose estimation (skipped sim)      │
│    MotionControlThread                      — control loop (both modes)          │
│    LiDARGetterThread                        — UDP lidar (skipped sim)            │
│                                                                                 │
│  _update_motion_control(state, lidar)       — L2MM predict → SocialNav step     │
│  _control_robot()                           — sport_client.Move(vx, vy, wz)     │
│  _show_results(image)                       — draw YOLO boxes, skeleton, BEV     │
│  run()                                      — Tkinter main loop + GUI updates    │
└──────────────┬──────────────────────────────────────────────────────────────────┘
               │ uses
               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MotionPredictor                                                                │
│  (models/api_language2mostion.py)                                               │
│                                                                                 │
│  __init__(model_path, tokenizer_path)       — loads L2MM transformer+tokenizer  │
│  predict(data) → {motion_vector, predicted_state, search_state}                 │
│  _preprocess_input(data) → input_ids, attention_mask                            │
│                                                                                 │
│  Inner model: LanguageToMotionTransformer                                       │
│    forward(input_ids, attn_mask) → motion_head [vx,vy,wz]                      │
│                                    mission_state_head, search_state_head         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  SequenceToSequenceClassAPI                                                     │
│  (models/api_object_extraction.py)                                              │
│                                                                                 │
│  __init__(model_path, tokenizer_path)       — loads extraction transformer      │
│  predict(instruction) → "handbag"           — extracts target class from text   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  SocialNavigator                                                                │
│  (models/api_social_navigator.py)                                               │
│                                                                                 │
│  __init__(enabled, **params)                — camera intrinsics, predictor, etc  │
│                                                                                 │
│  step(motion_vec, pose_state,               — full 8-stage pipeline             │
│       mission_state, lidar_ranges)            returns modulated velocity         │
│                                                                                 │
│  ── Stage 1: Detection ──                                                       │
│  _parse_pose_state(pose_state)              — YOLO keypoints+bboxes → det dicts │
│                                                                                 │
│  ── Stage 2: Distance Estimation ──                                             │
│  _project_lidar_to_image(lidar)             — 3D lidar → 2D image coords       │
│  _estimate_distances(dets, lidar)           — fuse lidar+mono, → position_rf    │
│  _estimate_distance_lidar(det, lidar)       — angular bbox query, cluster, med. │
│  _estimate_distance_mono(det)               — d = mono_k / bbox_height          │
│  _pixel_to_robot_frame(det)                 — pinhole: (u,v)+depth → [xlat, d]  │
│                                                                                 │
│  ── Stage 3: Tracking (ByteTrack) ──                                            │
│  _update_tracker(detections)                — 3-pass IoU assoc, lifecycle mgmt   │
│  _associate(dets, tracks, iou_thresh)       — Hungarian matching on IoU matrix   │
│  _compute_iou_matrix(boxes_a, boxes_b)      — pairwise IoU between bbox sets    │
│  _apply_detection(track, det, ts)           — copy det fields into track state   │
│                                                                                 │
│  ── Stage 4: Trajectory Prediction ──                                           │
│  _compensate_ego_motion()                   — rotate predictor history by turn   │
│  _predict_trajectories()                    — feed positions, predict, + ghosts  │
│    └─ ghost logic: out-of-FOV agents persist via extrapolated trajectory         │
│                                                                                 │
│  ── Stage 5: Safety Score ──                                                    │
│  _compute_safety_score()                    — calls safety.robot_safety_score()  │
│                                                                                 │
│  ── Stage 6: Shield Gate ──                                                     │
│  _evaluate_shield(mission_state)            — hysteresis: on < thresh_on,        │
│                                               off > thresh_off                   │
│                                                                                 │
│  ── Stage 7: Command Correction ──                                              │
│  _correct_command(motion_vector)            — potential-field omega correction    │
│  _potential_field_correction()              — grid gradient → angular steering   │
│                                                                                 │
│  ── Stage 8: Diagnostics ──                                                     │
│  _update_diagnostics()                      — log min_dist, humans, shield state │
│                                                                                 │
│  ── Visualization ──                                                            │
│  render_bev(show_heatmap)                   — BEV minimap: lidar, humans, ghosts │
│                                               trajectories, FOV, motion vectors  │
│  overlay_lidar(image)                       — draw lidar points on camera image  │
│  get_safety_heatmap(xlim, ylim, res)        — 2D safety grid for display        │
│  update_goal(object_xyn, bbox_h)            — set goal marker in robot frame    │
│  _extrapolate_robot_path(motion_vec)        — project robot path from velocity   │
└──────────────┬──────────────────────────────────────────────────────────────────┘
               │ uses
               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TrackedHuman                                                                   │
│  (models/api_social_navigator.py)           — data class, one per tracked person│
│                                                                                 │
│  Fields: track_id, position_image, bbox, keypoints, keypoints_conf, confidence  │
│          distance_lidar, distance_mono, distance, position_rf                   │
│          velocity, predicted_path, orientation, last_seen, is_ghost             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  HumanTrajectoryPredictor                                                       │
│  (models/humantrajectorypredictor.py)                                           │
│                                                                                 │
│  __init__(history_length, pred_steps, pred_interval)                            │
│  update_agent_position(id, [x,y], timestep) — append to sliding window          │
│  predict_trajectory(id) → [[x,y], ...]      — polyfit(1) linear extrapolation   │
│  predict_all(timestep) → {id: [[x,y],...]}   — predict all agents, throttled    │
│  prune_stale(active_ids)                     — remove gone agents               │
│  reset()                                     — clear all state                  │
│                                                                                 │
│  State: agent_trajectories = {id: deque([{position, timestep}, ...])}           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  safety.py (module-level functions)                                             │
│  (models/safety.py)                         — Gaussian + trajectory safety field│
│                                                                                 │
│  robot_safety_score(rx, ry, positions, paths) → float [0,1]                    │
│  safety_score_at_point(x, y, positions, paths) → float [0,1]                   │
│  compute_safety_grid(positions, xlim, ylim, res) → (grid, extent)              │
│  _gaussian_grid(X, Y, positions)            — proximity: exp(-d^2/2sigma^2)    │
│  _trajectory_grid(X, Y, paths)              — predicted path threat, gamma-dec  │
│  _safety_scores(X, Y, positions, paths)     — min(gaussian, trajectory)         │
│                                                                                 │
│  Constants: SIGMA=1.5m, H=1.0, GAMMA=0.995                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  CrowdNavDataProvider                                                           │
│  (models/crowdnav_data_provider.py)         — only used with --crowdnav_sim_mode│
│                                               replaces camera + lidar + YOLO    │
│                                                                                 │
│  __init__(env_config, policy_config, ...)   — creates CrowdSim env, ORCA humans│
│  reset(phase, test_case) → obs               — resets gym env                   │
│  step(motion_vector) → {pose_state, lidar, object_state, obs, reward, ...}     │
│  _generate_synthetic_pose_state(humans_rf)   — sim positions → fake YOLO dets   │
│  _generate_synthetic_lidar(humans_rf)        — sim positions → fake lidar cloud │
│  _generate_synthetic_object_state(self_st)   — goal → fake object detection     │
│  render_frame()                              — matplotlib top-down sim view     │
│                                                                                 │
│  Wraps: CrowdSim(gym.Env) with ORCA-controlled humans                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Ghost Humans

When a tracked human exits the camera FOV, their predictor history is retained and used to create "ghost" TrackedHumans (`is_ghost=True`) that follow the extrapolated trajectory. Ghosts maintain safety-grid influence and render as hollow circles on the BEV minimap. They expire after `ghost_max_frames` (default 120 frames / ~30s at 4 Hz) or when predicted behind the robot.

## Shield Hysteresis

The action shield uses two thresholds to prevent flickering:
- **Activate** when `safety_score < shield_thresh_on` (default 0.7)
- **Deactivate** when `safety_score > shield_thresh_off` (default 0.8)

---

## File Map

```
Social-LOVON/
├── deploy/
│   └── deploy.py                          # Entry point (real robot + sim mode)
├── models/
│   ├── api_social_navigator.py            # SocialNavigator + TrackedHuman
│   ├── api_language2mostion.py            # L2MM MotionPredictor
│   ├── api_object_extraction.py           # Object class extraction from text
│   ├── humantrajectorypredictor.py        # Linear trajectory extrapolation
│   ├── safety.py                          # Gaussian + trajectory safety field
│   └── crowdnav_data_provider.py          # Synthetic perception from CrowdNav sim
├── configs/
│   ├── env_lovon.config                   # CrowdSim environment parameters
│   └── policy_lovon.config                # Policy + kinematics config
├── crowd_sim/envs/                        # CrowdNav gym environment
│   ├── crowd_sim.py                       #   CrowdSim(gym.Env)
│   └── utils/                             #   Agent, Robot, Human, State, Action
└── crowd_nav/                             # CrowdNav policy registry + Explorer
```
