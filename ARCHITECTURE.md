# Social-LOVON Architecture

## Overview

Social-LOVON is a socially-aware robot navigation system. A language-to-motion transformer (L2MM) generates velocity commands from natural-language instructions, and a social navigation layer modulates those commands to avoid humans.

There are two execution paths: **simulation** (CrowdNav gym env) and **real robot** (Unitree Go2 via deploy.py). Both share the same SocialNavigator and L2MM core.

```
              crowd_test.py (sim)          deploy.py (robot)
                     |                           |
              LOVONCrowdPolicy          VisualLanguageController
                     |                           |
            +--------+--------+         +--------+--------+
            |                 |         |                 |
      MotionPredictor   SocialNavigator              SocialNavigator
       (L2MM model)     .step_ground_truth()          .step()
                              |                        |
                     TrajectoryPredictor        TrajectoryPredictor
                              |                        |
                          safety.py                safety.py
```

## Key Files

| File | Role |
|------|------|
| `deploy/deploy.py` | Real robot entry point. Multi-threaded: image capture, YOLO detection, YOLO pose, motion control, LiDAR. |
| `tools/crowd_test.py` | Simulation entry point. Single-threaded loop over CrowdNav gym env. |
| `models/lovon_crowd_policy.py` | CrowdNav policy wrapper. Bridges sim state into L2MM + SocialNavigator. |
| `models/api_language2mostion.py` | L2MM transformer. Text + object coords in, `[vx, vy, omega]` + state out. |
| `models/api_object_extraction.py` | Extracts target object class from instruction text. |
| `models/api_social_navigator.py` | SocialNavigator. Perception (YOLO→tracking→distance), trajectory prediction, safety scoring, action shield. |
| `models/humantrajectorypredictor.py` | Per-agent linear extrapolation from sliding-window history. |
| `models/safety.py` | Gaussian + trajectory safety score. Grid computation for heatmap. |
| `models/crowdnav_data_provider.py` | Generates synthetic YOLO-like detections from sim state for testing the full perception pipeline. |

## SocialNavigator Pipeline

Both `step()` (robot) and `step_ground_truth()` (sim) run stages 4-8. The robot path additionally runs stages 1-3.

| Stage | Method | What it does |
|-------|--------|--------------|
| 1 | `_parse_pose_state()` | YOLO keypoints + bboxes to detection dicts |
| 2 | `_estimate_distances()` | LiDAR depth (preferred) or monocular fallback, then pinhole→robot frame |
| 3 | `_update_tracker()` | ByteTrack IoU association: active → lost → pruned |
| 4 | `_predict_trajectories()` | Feed positions into predictor, extrapolate paths, create ghost humans for out-of-FOV persistence |
| 5 | `_compute_safety_score()` | Gaussian proximity + trajectory terms → score in [0, 1] |
| 6 | `_evaluate_shield()` | Hysteresis activation: on below `shield_thresh_on`, off above `shield_thresh_off` |
| 7 | `_correct_command()` | Potential-field velocity modulation when shield is active |
| 8 | `_update_diagnostics()` | Log and store min distance, human count, shield state |

## Ghost Humans

When a tracked human exits the camera FOV, the predictor history is retained and used to create "ghost" TrackedHumans that follow the extrapolated trajectory. Ghosts maintain safety-grid influence and render as hollow circles on the BEV minimap. They expire after `ghost_max_frames` (default 120 frames / ~30s) or when predicted behind the robot.

## Sim vs Robot

|  | Simulation | Robot |
|--|-----------|-------|
| Human detection | Ground-truth from CrowdSim | YOLO pose estimation |
| Distance | World-coord transform | LiDAR + monocular fusion |
| Tracking | Not needed (IDs given) | ByteTrack |
| Entry point | `step_ground_truth()` (skips stages 1-3) | `step()` (full pipeline) |
| Threading | Single-threaded | 5 threads (image, YOLO obj, YOLO pose, motion, LiDAR) |
| Actuation | `env.step(ActionXY/Rot)` | `sport_client.Move(vx, vy, wz)` |
