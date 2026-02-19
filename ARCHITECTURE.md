# Social-LOVON Architecture

## Overview

Social-LOVON is a socially-aware robot navigation system. A language-to-motion transformer (L2MM) generates velocity commands from natural-language instructions, and a social navigation layer modulates those commands to avoid humans.

Two execution paths share the same core: **simulation** (`crowd_test.py` + CrowdNav gym) and **real robot** (`deploy.py` + Unitree Go2).

---

## Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ENTRY POINTS                                                                    │
│                                                                                 │
│   tools/crowd_test.py (Simulation)             deploy/deploy.py (Real Robot)    │
│   └─ CrowdNav gym env + LOVONCrowdPolicy       └─ Unitree SDK + YOLO + VLC     │
└──────────────┬──────────────────────────────────────────────┬───────────────────┘
               │                                              │
               ▼                                              ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────────────┐
│  LOVONCrowdPolicy                │   │  VisualLanguageController (deploy.py)    │
│  (models/lovon_crowd_policy.py)  │   │                                          │
│  extends crowd_sim Policy        │   │  Threads:                                │
│                                  │   │   ImageGetterThread     (camera capture)  │
│  predict(state)                  │   │   YoloProcessingThread  (object det.)    │
│    1. _build_l2mm_input()        │   │   YoloPoseProcessingThread (pose est.)   │
│    2. _call_l2mm()               │   │   MotionControlThread   (control loop)   │
│    3. _call_social_nav()         │   │   LiDARGetterThread     (UDP lidar)      │
│    4. _to_action()               │   │                                          │
│                                  │   │  _update_motion_control(state)            │
│  load_lovon(model, tok, social)  │   │    1. motion_predictor.predict()          │
│  set_mission(instr0, instr1, obj)│   │    2. social_nav.step()                  │
│  _sim_humans_to_robot_frame()    │   │    3. _control_robot() → Move(vx,vy,wz)  │
│  _angle_to_xyn_whn()            │   │                                          │
│  world_to_robot_frame()          │   │                                          │
│  world_to_robot_frame_velocity() │   │                                          │
└───────────┬──────────────────────┘   └──────────┬───────────────────────────────┘
            │                                      │
            │  uses                                │  uses
            ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MotionPredictor                                                                │
│  (models/api_language2mostion.py)                                               │
│                                                                                 │
│  __init__(model_path, tokenizer_path)   — loads L2MM transformer + tokenizer    │
│  predict(data) → {motion_vector, predicted_state, search_state}                 │
│  _preprocess_input(data) → input_ids, attention_mask                            │
│                                                                                 │
│  Inner model: LanguageToMotionTransformer                                       │
│    forward(input_ids, attn_mask) → motion_head [vx,vy,wz]                      │
│                                    mission_state_head, search_state_head         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  SequenceToSequenceClassAPI                                                     │
│  (models/api_object_extraction.py)         — used by deploy.py only             │
│                                                                                 │
│  __init__(model_path, tokenizer_path)   — loads extraction transformer          │
│  predict(instruction) → "handbag"       — extracts target class from text       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  SocialNavigator                                                                │
│  (models/api_social_navigator.py)                                               │
│                                                                                 │
│  __init__(enabled, **params)            — camera intrinsics, predictor, tracker  │
│                                                                                 │
│  step(motion_vec, pose_state,           — FULL pipeline (robot path)            │
│       mission_state, lidar_ranges)        stages 1-8, returns modulated vec     │
│  step_ground_truth(motion_vec,          — SIM pipeline (skips stages 1-3)       │
│       gt_humans, mission_state)           populates humans directly from sim    │
│                                                                                 │
│  ── Stage 1: Detection ──                                                       │
│  _parse_pose_state(pose_state)          — YOLO keypoints+bboxes → det dicts     │
│                                                                                 │
│  ── Stage 2: Distance Estimation ──                                             │
│  _project_lidar_to_image(lidar)         — 3D lidar → 2D image coordinates       │
│  _estimate_distances(dets, lidar)       — fuse lidar+mono, compute position_rf  │
│  _estimate_distance_lidar(det, lidar)   — angular bbox query, cluster, median   │
│  _estimate_distance_mono(det)           — d = mono_k / bbox_height              │
│  _pixel_to_robot_frame(det)             — pinhole: (u,v)+depth → [x_lat, depth] │
│                                                                                 │
│  ── Stage 3: Tracking (ByteTrack) ──                                            │
│  _update_tracker(detections)            — 3-pass IoU association, lifecycle mgmt │
│  _associate(dets, tracks, iou_thresh)   — Hungarian matching on IoU matrix       │
│  _compute_iou_matrix(boxes_a, boxes_b)  — pairwise IoU between two bbox sets    │
│  _apply_detection(track, det, ts)       — copy det fields into track state       │
│                                                                                 │
│  ── Stage 4: Trajectory Prediction ──                                           │
│  _compensate_ego_motion()               — rotate predictor history by robot turn │
│  _predict_trajectories()                — feed positions, predict, create ghosts │
│    └─ ghost logic: persist out-of-FOV agents via extrapolated trajectory         │
│                                                                                 │
│  ── Stage 5: Safety Score ──                                                    │
│  _compute_safety_score()                — calls safety.robot_safety_score()      │
│                                                                                 │
│  ── Stage 6: Shield Gate ──                                                     │
│  _evaluate_shield(mission_state)        — hysteresis: on < thresh_on,            │
│                                           off > thresh_off                       │
│                                                                                 │
│  ── Stage 7: Command Correction ──                                              │
│  _correct_command(motion_vector)        — potential-field omega correction        │
│  _potential_field_correction()           — grid gradient → angular steering       │
│                                                                                 │
│  ── Stage 8: Diagnostics ──                                                     │
│  _update_diagnostics()                  — log min_dist, num_humans, shield state │
│                                                                                 │
│  ── Visualization ──                                                            │
│  render_bev(show_heatmap)               — BEV minimap: lidar, humans, ghosts,   │
│                                           trajectories, FOV, motion vectors      │
│  overlay_lidar(image)                   — draw lidar points on camera image      │
│  get_safety_heatmap(xlim, ylim, res)    — compute 2D safety grid for display    │
│  update_goal(object_xyn, bbox_h)        — set goal marker in robot frame        │
│  _extrapolate_robot_path(motion_vec)    — project robot path from velocity       │
└──────────────┬──────────────────────────────────────────────────────────────────┘
               │ uses
               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  HumanTrajectoryPredictor                                                       │
│  (models/humantrajectorypredictor.py)                                           │
│                                                                                 │
│  __init__(history_length, pred_steps, pred_interval)                            │
│  update_agent_position(agent_id, [x,y], timestep) — append to sliding window    │
│  predict_trajectory(agent_id) → [[x,y], ...]      — polyfit(1) extrapolation    │
│  predict_all(timestep) → {id: [[x,y],...]}         — predict all, throttled     │
│  prune_stale(active_ids)                           — remove gone agents          │
│  reset()                                           — clear all state             │
│                                                                                 │
│  State: agent_trajectories = {id: deque([{position, timestep}, ...])}           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  TrackedHuman                                                                   │
│  (models/api_social_navigator.py)       — data class, one per tracked person    │
│                                                                                 │
│  Fields: track_id, position_image, bbox, keypoints, keypoints_conf, confidence  │
│          distance_lidar, distance_mono, distance, position_rf                   │
│          velocity, predicted_path, orientation, last_seen, is_ghost             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  safety.py (module-level functions)                                             │
│  (models/safety.py)                     — Gaussian + trajectory safety field    │
│                                                                                 │
│  robot_safety_score(rx, ry, positions, paths) → float [0,1]                    │
│  safety_score_at_point(x, y, positions, paths) → float [0,1]                   │
│  compute_safety_grid(positions, xlim, ylim, res) → (grid, extent)              │
│  _gaussian_grid(X, Y, positions)        — proximity: exp(-d^2 / 2sigma^2)      │
│  _trajectory_grid(X, Y, paths)          — predicted path threat, gamma-decayed  │
│  _safety_scores(X, Y, positions, paths) — min(gaussian, trajectory), clipped    │
│                                                                                 │
│  Constants: SIGMA=1.5m, H=1.0, GAMMA=0.995                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  CrowdNavDataProvider                                                           │
│  (models/crowdnav_data_provider.py)     — bridges CrowdNav sim → deploy.py      │
│                                           perception format (--crowdnav_sim_mode)│
│                                                                                 │
│  __init__(env_config, policy_config, ...) — creates CrowdSim env + ORCA humans  │
│  reset(phase, test_case) → obs           — resets env, returns initial obs       │
│  step(motion_vector) → (obs, reward, done, info)                                │
│  _generate_synthetic_pose_state(humans)  — sim positions → fake YOLO detections  │
│  _generate_synthetic_lidar(humans)       — sim positions → fake lidar cloud      │
│  _generate_synthetic_object_state(self_state) — goal → fake object detection     │
│  render_frame()                          — matplotlib top-down sim visualization │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Sim vs Robot

|  | Simulation (`crowd_test.py`) | Robot (`deploy.py`) |
|--|------------------------------|---------------------|
| Human detection | Ground-truth from CrowdSim | YOLO pose estimation |
| Distance | World-coord transform | LiDAR + monocular fusion |
| Tracking | IDs given by sim | ByteTrack IoU association |
| SocialNav entry | `step_ground_truth()` (stages 1-3 skipped) | `step()` (full pipeline) |
| Threading | Single-threaded | 5 threads |
| Actuation | `env.step(ActionXY/Rot)` | `sport_client.Move(vx, vy, wz)` |

There is also a hybrid mode (`deploy.py --crowdnav_sim_mode`) that uses CrowdNavDataProvider to generate synthetic YOLO/lidar from the sim, exercising the full perception pipeline without a real robot.

## Ghost Humans

When a tracked human exits the camera FOV, their predictor history is retained and used to create "ghost" TrackedHumans (`is_ghost=True`) that follow the extrapolated trajectory. Ghosts maintain safety-grid influence and render as hollow circles on the BEV minimap. They expire after `ghost_max_frames` (default 120 / ~30s at 4 Hz) or when predicted behind the robot.

## Shield Hysteresis

The action shield uses two thresholds to prevent flickering:
- **Activate** when `safety_score < shield_thresh_on` (default 0.7)
- **Deactivate** when `safety_score > shield_thresh_off` (default 0.8)

## File Map

```
Social-LOVON/
├── deploy/
│   └── deploy.py                          # Real robot entry point
├── tools/
│   ├── crowd_test.py                      # Simulation entry point
│   └── crowd_test_live.py                 # Live visualization variant
├── models/
│   ├── api_social_navigator.py            # SocialNavigator + TrackedHuman
│   ├── api_language2mostion.py            # L2MM MotionPredictor
│   ├── api_object_extraction.py           # Object extraction from text
│   ├── humantrajectorypredictor.py        # Linear trajectory extrapolation
│   ├── safety.py                          # Gaussian + trajectory safety
│   ├── lovon_crowd_policy.py              # CrowdNav policy wrapper
│   └── crowdnav_data_provider.py          # Synthetic perception from sim
├── configs/
│   ├── env_lovon.config                   # CrowdSim env parameters
│   └── policy_lovon.config                # Policy + kinematics config
├── crowd_sim/envs/                        # CrowdNav gym environment
│   ├── crowd_sim.py                       #   CrowdSim(gym.Env)
│   └── utils/                             #   Agent, Robot, Human, State, Action
└── crowd_nav/                             # CrowdNav policy registry + Explorer
```
