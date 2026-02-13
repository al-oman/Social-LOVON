# Social-LOVON Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                    │
│                                                                         │
│   crowd_test.py (Simulation)              deploy.py (Real Robot)        │
│   └─ CrowdNav gym environment             └─ Unitree SDK + YOLO +      │
│      + LOVONCrowdPolicy                      VisualLanguageController   │
└─────────────────┬───────────────────────────────────┬───────────────────┘
                  │                                   │
                  ▼                                   ▼
┌─────────────────────────────┐   ┌───────────────────────────────────────┐
│   models/                   │   │   models/                             │
│   ┌───────────────────────┐ │   │   ┌─────────────────────────────────┐ │
│   │ lovon_crowd_policy.py │ │   │   │ api_object_extraction.py       │ │
│   │  (Policy subclass)    │ │   │   │  (extract target from text)    │ │
│   └──────────┬────────────┘ │   │   └─────────────────────────────────┘ │
│              │              │   │   ┌─────────────────────────────────┐ │
│   ┌──────────▼────────────┐ │   │   │ api_language2mostion.py        │ │
│   │ api_language2mostion  │◄├───┤   │  (L2MM Transformer)            │ │
│   │  (MotionPredictor)    │ │   │   └─────────────────────────────────┘ │
│   └──────────┬────────────┘ │   │   ┌─────────────────────────────────┐ │
│              │              │   │   │ api_social_navigator.py        │ │
│   ┌──────────▼────────────┐ │   │   │  (SocialNavigator)             │ │
│   │ api_social_navigator  │◄├───┤   └──────────┬──────────────────────┘ │
│   │ (SocialNavigator)     │ │   │              │                        │
│   └──────────┬────────────┘ │   │   ┌──────────▼──────────────────────┐ │
│              │              │   │   │ humantrajectorypredictor.py    │ │
│   ┌──────────▼────────────┐ │   │   │  (linear extrapolation)       │ │
│   │ safety.py             │ │   │   └─────────────────────────────────┘ │
│   │ (Gaussian + traj)     │ │   │   ┌─────────────────────────────────┐ │
│   └───────────────────────┘ │   │   │ safety.py                      │ │
│                             │   │   │  (robot_safety_score)           │ │
└─────────────────────────────┘   │   └─────────────────────────────────┘ │
                                  └───────────────────────────────────────┘
```

---

## A. SIMULATION PATH — `crowd_test.py`

### Startup Sequence

```
crowd_test.py main()
│
├─1─ policy_factory['lovon'] = LOVONCrowdPolicy
│
├─2─ policy = policy_factory['lovon']()          # LOVONCrowdPolicy.__init__()
│    └─ sets kinematics="unicycle", trainable=False
│
├─3─ policy.configure(policy_config)             # reads [lovon].kinematics from .ini
│
├─4─ policy.load_lovon(model_path, tokenizer_path, social_nav_enabled=True)
│    ├─ self.l2mm = MotionPredictor(model_path, tokenizer_path)
│    │   └─ loads LanguageToMotionTransformer weights + tokenizer
│    └─ self.social_nav = SocialNavigator(enabled=True)
│        └─ self._predictor = HumanTrajectoryPredictor()
│
├─5─ policy.set_mission("run to bag at 1.0 m/s", "run to bag at 1.0 m/s", "handbag")
│    └─ stores mission_instruction_0/1, predicted_object
│
├─6─ env = gym.make('CrowdSim-v0')              # CrowdSim from crowd_sim
│    env.configure(env_config)                    # time_step, human_num, rewards
│
├─7─ env.safety_calculator = lambda: compute_safety_grid(...)   # INJECTED
│
├─8─ robot = Robot(env_config, 'robot')
│    robot.set_policy(policy)                     # robot.policy = LOVONCrowdPolicy
│    env.set_robot(robot)
│
└─9─ explorer = Explorer(env, robot, device, gamma=0.9)
     └─ explorer.run_k_episodes(...)  OR  manual loop with env.reset/step
```

### Per-Timestep Call Chain (Simulation)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ robot.act(ob)         ob = [ObservableState, ...]  (one per human)          │
│ │                                                                            │
│ ├─ state = JointState(                                                       │
│ │     self_state = robot.get_full_state(),   ← FullState(px,py,vx,vy,       │
│ │     human_states = ob                         radius,gx,gy,v_pref,theta)  │
│ │   )                                                                        │
│ │                                                                            │
│ └─ policy.predict(state)                     ← LOVONCrowdPolicy.predict()   │
│    │                                                                         │
│    ├─── _build_l2mm_input(self_state, human_states) ────────────────────┐    │
│    │    │                                                                │    │
│    │    ├─ _angle_to_xyn_whn(self_state)                                │    │
│    │    │   └─ computes goal direction in image coordinates              │    │
│    │    │   └─ returns (xyn, whn) normalized pixel coords               │    │
│    │    │                                                                │    │
│    │    └─ returns dict ─────────────────────────────────────────────────┘    │
│    │        mission_instruction_0: str                                        │
│    │        mission_instruction_1: str                                        │
│    │        predicted_object: str (e.g. "handbag")                           │
│    │        confidence: [float]                                              │
│    │        object_xyn: [norm_x, norm_y]                                     │
│    │        object_whn: [norm_w, norm_h]                                     │
│    │        mission_state_in: "running"|"searching_0"|"searching_1"|"success"│
│    │        search_state_in: "had_searching_0"|"had_searching_1"             │
│    │                                                                         │
│    ├─── _call_l2mm(l2mm_input) ─────────────────────────────────────────┐    │
│    │    │                                                                │    │
│    │    └─ self.l2mm.predict(l2mm_input)     ← MotionPredictor.predict  │    │
│    │       ├─ _preprocess_input(data)                                    │    │
│    │       │   └─ builds text string from all fields                     │    │
│    │       │   └─ tokenizes → input_ids + attention_mask                 │    │
│    │       └─ model.forward(input_ids, attention_mask)                    │    │
│    │           ├─ motion_head → [vx, vy, omega_z]                        │    │
│    │           ├─ mission_state_head → argmax → state string             │    │
│    │           └─ search_state_head → argmax → state string              │    │
│    │                                                                     │    │
│    │    returns dict ────────────────────────────────────────────────────┘    │
│    │      motion_vector: [vx, vy, omega_z]                                   │
│    │      predicted_state: str                                               │
│    │      search_state: str                                                  │
│    │                                                                         │
│    ├─── _call_social_nav(motion_vector, self_state, human_states) ──────┐    │
│    │    │                                                                │    │
│    │    ├─ _sim_humans_to_robot_frame(self_state, human_states)          │    │
│    │    │   └─ for each human:                                           │    │
│    │    │       world_to_robot_frame(h.px, h.py, r.px, r.py, r.theta)   │    │
│    │    │       world_to_robot_frame_velocity(h.vx, h.vy, r.theta)      │    │
│    │    │   └─ returns gt_humans: [{track_id, position_rf, distance,    │    │
│    │    │                           velocity, radius}, ...]              │    │
│    │    │                                                                │    │
│    │    └─ self.social_nav.step_ground_truth(motion_vec, gt_humans,      │    │
│    │       │                                  mission_state)             │    │
│    │       │                                                             │    │
│    │       │  ┌── SocialNavigator.step_ground_truth() ──────────────┐   │    │
│    │       │  │                                                      │   │    │
│    │       │  │  1. Populate _tracked_humans directly from gt_humans │   │    │
│    │       │  │     (BYPASSES detection + distance + tracking)       │   │    │
│    │       │  │                                                      │   │    │
│    │       │  │  2. _predict_trajectories()                          │   │    │
│    │       │  │     └─ HumanTrajectoryPredictor.update_agent_position│   │    │
│    │       │  │     └─ HumanTrajectoryPredictor.predict_all()        │   │    │
│    │       │  │        └─ np.polyfit linear extrapolation per human  │   │    │
│    │       │  │                                                      │   │    │
│    │       │  │  3. _compute_safety_score()                          │   │    │
│    │       │  │     └─ safety.robot_safety_score(0, 0, positions,    │   │    │
│    │       │  │        │                          predicted_paths)    │   │    │
│    │       │  │        └─ safety_score_at_point()                    │   │    │
│    │       │  │           ├─ _gaussian_term(robot, humans)           │   │    │
│    │       │  │           └─ _trajectory_term(robot, paths)          │   │    │
│    │       │  │           └─ min(s_gauss, s_traj) ∈ [0,1]           │   │    │
│    │       │  │                                                      │   │    │
│    │       │  │  4. _evaluate_shield(mission_state)                  │   │    │
│    │       │  │     └─ shield_active = score < thresh                │   │    │
│    │       │  │        AND state in shield_active_states             │   │    │
│    │       │  │                                                      │   │    │
│    │       │  │  5. _correct_command(motion_vector)                  │   │    │
│    │       │  │     └─ velocity modulation if shield active          │   │    │
│    │       │  │                                                      │   │    │
│    │       │  │  6. _update_diagnostics()                            │   │    │
│    │       │  │                                                      │   │    │
│    │       │  └──────────────────────────────────────────────────────┘   │    │
│    │       │                                                             │    │
│    │    returns: motion_vector (possibly modulated) ─────────────────────┘    │
│    │                                                                         │
│    └─── _to_action(motion_vector, self_state) ──────────────────────────┐    │
│         │                                                                │    │
│         ├─ if holonomic: ActionXY(vx, vy)                                │    │
│         └─ if unicycle:  ActionRot(speed, rotation_per_step)             │    │
│                                                                          │    │
│         returns: ActionXY | ActionRot ───────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ env.step(action)          CrowdSim.step()                                    │
│ │                                                                            │
│ ├─ for each human:                                                           │
│ │   human_ob = [other humans' + robot's ObservableState]                     │
│ │   human_action = human.act(human_ob)                                       │
│ │   └─ human.policy.predict(JointState)   ← typically ORCA policy            │
│ │                                                                            │
│ ├─ collision detection (robot vs each human, pairwise distances)             │
│ ├─ goal-reached check (robot within goal_radius of (gx, gy))                │
│ ├─ reward computation:                                                       │
│ │   ├─ collision: -0.25                                                      │
│ │   ├─ reaching_goal: +1.0                                                   │
│ │   ├─ discomfort (within discomfort_dist): penalty                          │
│ │   └─ otherwise: 0                                                          │
│ │                                                                            │
│ ├─ robot.step(action)        ← updates robot px,py,vx,vy,theta              │
│ ├─ human.step(human_action)  ← updates each human px,py,vx,vy               │
│ │                                                                            │
│ └─ returns (ob, reward, done, info)                                          │
│     ob = [human.get_observable_state() for human in humans]                  │
│     info = ReachGoal | Collision | Timeout | Danger | Nothing                │
│                                                                              │
│ ── if --safety_heatmap: ──                                                   │
│    env.safety_calculator(human_states, xlim, ylim)                           │
│    └─ compute_safety_grid(human_positions, xlim, ylim)     ← safety.py       │
│       └─ for each grid cell: safety_score_at_point()                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## B. REAL ROBOT PATH — `deploy.py`

### Thread Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VisualLanguageController                                │
│                                                                             │
│  ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐ │
│  │ LiDARGetterThread│     │ ImageGetterThread    │     │  Tkinter UI      │ │
│  │                  │     │                      │     │  Thread (main)   │ │
│  │ polls UDP topic  │     │ webcam / Go2 camera  │     │                  │ │
│  │ 'rt/utlidar/     │     │ / RealSense          │     │ update_image()   │ │
│  │  cloud'          │     │                      │     │ every 100ms      │ │
│  │                  │     │ blur detection        │     │                  │ │
│  │ latest_cloud ────┼──┐  │ (Laplacian variance) │     │ _show_results()  │ │
│  │ {x,y,z} arrays  │  │  │                      │     │ └─ draw YOLO box │ │
│  └──────────────────┘  │  ├──────────┬───────────┤     │ └─ draw skeleton │ │
│                        │  │          │           │     │ └─ draw BEV      │ │
│                        │  │   image_queue  pose_image  │ └─ social_nav    │ │
│                        │  │          │     _queue │     │    .draw_bev()   │ │
│                        │  │          ▼           ▼     │                  │ │
│                        │  │  ┌────────────┐ ┌────────┐│                  │ │
│                        │  │  │YoloProcess-│ │YoloPose││                  │ │
│                        │  │  │ingThread   │ │Process- ││                  │ │
│                        │  │  │            │ │ingThrd  ││                  │ │
│                        │  │  │YOLO obj    │ │YOLO pose││                  │ │
│                        │  │  │detection   │ │estimat. ││                  │ │
│                        │  │  │            │ │         ││                  │ │
│                        │  │  │_yolo_image │ │_yolo_   ││                  │ │
│                        │  │  │_post_proc()│ │pose_    ││                  │ │
│                        │  │  │            │ │post_    ││                  │ │
│                        │  │  │Updates:    │ │proc()   ││                  │ │
│                        │  │  │ state{     │ │         ││                  │ │
│                        │  │  │  predicted │ │Updates: ││                  │ │
│                        │  │  │  _object,  │ │pose_    ││                  │ │
│                        │  │  │  confidence│ │state{   ││                  │ │
│                        │  │  │  object_xyn│ │num_ppl, ││                  │ │
│                        │  │  │  object_whn│ │poses,   ││                  │ │
│                        │  │  │ }          │ │boxes    ││                  │ │
│                        │  │  │            │ │}        ││                  │ │
│                        │  │  └─────┬──────┘ └───┬────┘│                  │ │
│                        │  │        │            │     │                  │ │
│                        │  │        ▼            │     │                  │ │
│                        │  │  ┌─────────────────────┐  │                  │ │
│                        │  │  │ MotionControlThread  │  │                  │ │
│                        │  │  │                      │  │                  │ │
│                        │  │  │ ┌──────────────────┐ │  │                  │ │
│                        └──┼──┼─► _update_motion_  │ │  │                  │ │
│                           │  │ │  control(state)  │ │  │                  │ │
│                           │  │ │                  │ │  │                  │ │
│                           │  │ │  (see below)     │ │  │                  │ │
│                           │  │ └────────┬─────────┘ │  │                  │ │
│                           │  │          │           │  │                  │ │
│                           │  │          ▼           │  │                  │ │
│                           │  │ ┌──────────────────┐ │  │                  │ │
│                           │  │ │ _control_robot() │ │  │                  │ │
│                           │  │ │ sport_client     │ │  │                  │ │
│                           │  │ │ .Move(vx,vy,wz)  │ │  │                  │ │
│                           │  │ └──────────────────┘ │  │                  │ │
│                           │  └──────────────────────┘  │                  │ │
│                           └────────────────────────────┘                  │ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### `_update_motion_control(state)` Detail (Deploy Path)

```
_update_motion_control(state)
│
├─ input_data = {
│    "mission_instruction_0": self.mission_instruction_0,
│    "mission_instruction_1": self.mission_instruction_1,
│    **state    ← predicted_object, confidence, object_xyn, object_whn,
│                  mission_state_in, search_state_in
│  }
│
├─ self.motion_predictor.predict(input_data)        ← MotionPredictor
│   └─ returns: motion_vector, predicted_state, search_state
│
├─ update self.state with new predicted_state, search_state
│
├─ lidar_cloud = self.lidar_getter_thread.get_cloud()
│
└─ self.motion_vector = self.social_nav.step(       ← SocialNavigator.step()
     motion_vector, pose_state, mission_state, lidar_cloud)
```

### `SocialNavigator.step()` (Real Robot — Full Pipeline)

```
SocialNavigator.step(motion_vector, pose_state, mission_state, lidar_ranges)
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ STAGES 1-3: PERCEPTION  (skipped in step_ground_truth)         │
│  │                                                                 │
│  │ 1. _parse_pose_state(pose_state)                               │
│  │    └─ converts YOLO pose keypoints + bboxes → detection dicts  │
│  │       [{bbox, keypoints, confidence}, ...]                     │
│  │                                                                 │
│  │ 2. _estimate_distances(detections, lidar_ranges)               │
│  │    ├─ _estimate_distance_lidar(det, lidar_ranges)              │
│  │    │   └─ project bbox to angular range, query LiDAR points,   │
│  │    │      return median distance                                │
│  │    ├─ _estimate_distance_mono(det)                             │
│  │    │   └─ d ≈ mono_k / bbox_height_px                          │
│  │    ├─ fuse: prefer lidar, fallback to mono                     │
│  │    └─ _pixel_to_robot_frame(det)                               │
│  │        └─ pinhole model: (u,v) + depth → [x_lateral, depth]   │
│  │                                                                 │
│  │ 3. _update_tracker(detections)                                 │
│  │    └─ ByteTrack-style IoU association                          │
│  │       ├─ split detections by confidence threshold              │
│  │       ├─ 3-pass association (high→active, low→remaining,       │
│  │       │                      high→lost)                        │
│  │       ├─ manage lifecycle: active → lost → pruned              │
│  │       └─ populate _tracked_humans from active tracks           │
│  │                                                                 │
│  └─────────────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ STAGES 4-8: PREDICTION + SAFETY  (shared with step_ground_truth)
│  │                                                                 │
│  │ 4. _predict_trajectories()                                     │
│  │    └─ HumanTrajectoryPredictor                                 │
│  │       ├─ update_agent_position(id, pos, timestep)              │
│  │       └─ predict_all(timestep)                                 │
│  │           └─ linear polynomial extrapolation per agent         │
│  │                                                                 │
│  │ 5. _compute_safety_score()                                     │
│  │    └─ safety.robot_safety_score(0, 0, human_positions,         │
│  │       │                          predicted_paths)               │
│  │       └─ safety_score_at_point(0, 0, ...)                      │
│  │           ├─ _gaussian_term()  ← proximity danger              │
│  │           └─ _trajectory_term() ← predicted path danger        │
│  │           └─ returns min(gauss, traj) ∈ [0.0, 1.0]            │
│  │                                                                 │
│  │ 6. _evaluate_shield(mission_state)                             │
│  │    └─ shield_active = (score < threshold)                      │
│  │       AND (mission_state in active_states)                     │
│  │                                                                 │
│  │ 7. _correct_command(motion_vector)                             │
│  │    └─ if shield_active: modulate velocities                    │
│  │                                                                 │
│  │ 8. _update_diagnostics()                                       │
│  │    └─ min_distance, num_humans, shield_active, safety_score    │
│  │                                                                 │
│  └─────────────────────────────────────────────────────────────────┘
│
└─ returns: motion_vector (possibly modulated)
```

---

## C. CROWDNAV CLASSES (crowd_sim / crowd_nav)

```
crowd_sim/envs/
├── crowd_sim.py ─────────── CrowdSim(gym.Env)
│                              ├─ configure(config)
│                              ├─ reset(phase, test_case) → [ObservableState]
│                              ├─ step(action) → (ob, reward, done, info)
│                              ├─ generate_random_human_position()
│                              └─ safety_calculator (injected by crowd_test.py)
│
├── utils/
│   ├── state.py ──────────── FullState(px,py,vx,vy,radius,gx,gy,v_pref,theta)
│   │                         ObservableState(px,py,vx,vy,radius)
│   │                         JointState(self_state, human_states)
│   │
│   ├── action.py ─────────── ActionXY(vx, vy)
│   │                         ActionRot(v, r)
│   │
│   ├── agent.py ──────────── Agent (base)
│   │                           ├─ get_full_state() → FullState
│   │                           ├─ get_observable_state() → ObservableState
│   │                           ├─ set(px,py,gx,gy,vx,vy,theta)
│   │                           └─ step(action) / compute_position(action, dt)
│   │
│   ├── robot.py ──────────── Robot(Agent)
│   │                           └─ act(ob) → policy.predict(JointState)
│   │
│   └── human.py ─────────── Human(Agent)
│                               └─ act(ob) → policy.predict(JointState)
│
└── policy/
    ├── policy.py ─────────── Policy (abstract base)
    │                           ├─ configure(config)
    │                           └─ predict(state) → Action
    │
    └── orca.py ───────────── ORCA(Policy)
                                └─ predict(state) → ActionXY
                                   (used for human agents in sim)

crowd_nav/
├── policy/
│   └── policy_factory.py ── policy_factory: dict  {'orca': ORCA, ...}
│                            crowd_test.py adds: policy_factory['lovon'] = LOVONCrowdPolicy
│
└── utils/
    └── explorer.py ──────── Explorer(env, robot, device, gamma)
                               └─ run_k_episodes(k, phase)
                                  └─ loops: robot.act(ob) → env.step(action)
```

---

## D. DATA FLOW — Values Passed at Each Boundary

### 1. crowd_test.py → LOVONCrowdPolicy

```
policy.predict(state: JointState)
  state.self_state = FullState:
    ├─ px, py        robot world position
    ├─ vx, vy        robot world velocity
    ├─ radius         robot radius
    ├─ gx, gy        goal world position
    ├─ v_pref        preferred speed
    └─ theta          robot heading (rad)

  state.human_states = [ObservableState]:
    ├─ px, py        human world position
    ├─ vx, vy        human world velocity
    └─ radius         human radius
```

### 2. LOVONCrowdPolicy → MotionPredictor

```
l2mm.predict(input_data: dict)
  ├─ mission_instruction_0: str      "run to bag at 1.0 m/s"
  ├─ mission_instruction_1: str      "run to bag at 1.0 m/s"
  ├─ predicted_object: str           "handbag"
  ├─ confidence: [float]             [0.85]
  ├─ object_xyn: [float, float]     normalized image x,y of target
  ├─ object_whn: [float, float]     normalized bbox w,h of target
  ├─ mission_state_in: str          "running"
  └─ search_state_in: str           "had_searching_0"

Returns:
  ├─ motion_vector: [vx, vy, wz]    commanded velocities
  ├─ predicted_state: str            new mission state
  └─ search_state: str               new search state
```

### 3. LOVONCrowdPolicy → SocialNavigator.step_ground_truth()

```
social_nav.step_ground_truth(motion_vector, gt_humans, mission_state)
  motion_vector: [vx, vy, wz]
  gt_humans: [dict]:
    ├─ track_id: int
    ├─ position_rf: [x_lateral, depth]    robot-frame position
    ├─ distance: float                     Euclidean distance
    ├─ velocity: [vx_lat, v_depth]        robot-frame velocity
    └─ radius: float
  mission_state: str                       "running"|"success"|...

Returns:
  └─ motion_vector: [vx, vy, wz]          possibly modulated
```

### 4. deploy.py → SocialNavigator.step()

```
social_nav.step(motion_vector, pose_state, mission_state, lidar_ranges)
  motion_vector: [vx, vy, wz]
  pose_state: dict:
    ├─ num_people: int
    ├─ poses: [[17 keypoints × (x,y,conf)], ...]
    └─ pose_boxes: [[x1,y1,x2,y2,conf], ...]
  mission_state: str
  lidar_ranges: dict {x: ndarray, y: ndarray, z: ndarray} | None

Returns:
  └─ motion_vector: [vx, vy, wz]          possibly modulated
```

### 5. SocialNavigator → safety.py

```
robot_safety_score(robot_x, robot_y, human_positions, human_predicted_paths)
  robot_x, robot_y: float             always (0, 0) in robot frame
  human_positions: [(x,y), ...]       robot-frame positions
  human_predicted_paths: {id: [[x,y], ...]}  predicted trajectories

Returns:
  └─ float ∈ [0.0, 1.0]              1.0=safe, 0.0=dangerous
```

### 6. crowd_test.py → safety.py (heatmap visualization)

```
compute_safety_grid(human_positions, xlim, ylim, resolution=0.1)
  human_positions: [(px,py), ...]      world-frame positions
  xlim: (float, float)                 grid x bounds
  ylim: (float, float)                 grid y bounds

Returns:
  ├─ grid: 2D ndarray                 safety values per cell
  └─ extent: (xmin, xmax, ymin, ymax) for matplotlib imshow
```

### 7. deploy.py → SequenceToSequenceClassAPI (object extraction)

```
object_extractor.predict(mission_instruction_1: str)
  Input:  "run to the bag on the left"
  Output: "handbag"   (class name from class_mapping.json)
```

---

## E. COMPLETE FILE DEPENDENCY GRAPH

```
crowd_test.py
  ├── imports ──► models/lovon_crowd_policy.py ─┬──► models/api_language2mostion.py
  │               (LOVONCrowdPolicy)             │    (MotionPredictor)
  │                                              │      └──► LanguageToMotionTransformer
  │                                              │
  │                                              ├──► models/api_social_navigator.py
  │                                              │    (SocialNavigator)
  │                                              │      ├──► models/humantrajectorypredictor.py
  │                                              │      │    (HumanTrajectoryPredictor)
  │                                              │      └──► models/safety.py
  │                                              │           (robot_safety_score)
  │                                              │
  │                                              └──► crowd_sim/.../policy.py (Policy base)
  │                                                   crowd_sim/.../action.py (ActionXY, ActionRot)
  │
  ├── imports ──► models/safety.py
  │               (compute_safety_grid)
  │
  ├── imports ──► crowd_sim/envs/crowd_sim.py (CrowdSim via gym.make)
  ├── imports ──► crowd_sim/envs/utils/robot.py (Robot)
  ├── imports ──► crowd_sim/envs/policy/orca.py (ORCA)
  └── imports ──► crowd_nav/utils/explorer.py (Explorer)

deploy.py
  ├── imports ──► models/api_language2mostion.py (MotionPredictor)
  ├── imports ──► models/api_object_extraction.py (SequenceToSequenceClassAPI)
  │                └──► SequenceToSequenceClassTransformer
  ├── imports ──► models/api_social_navigator.py (SocialNavigator)
  │                ├──► models/humantrajectorypredictor.py
  │                └──► models/safety.py
  └── imports ──► ultralytics (YOLO)
  └── imports ──► unitree_sdk2py (robot control)
```

---

## F. KEY DIFFERENCE: SIMULATION vs DEPLOY

```
                        SIMULATION (crowd_test)     DEPLOY (deploy.py)
                        ────────────────────────    ──────────────────────
Humans detected by:     CrowdSim ground truth       YOLO pose estimation
                        (ObservableState)            (keypoints + bboxes)

Distance estimated:     Known (world coords)        LiDAR + monocular
                        direct transform             fusion

Tracking:               Not needed (IDs given)       ByteTrack IoU assoc.

SocialNav entry:        step_ground_truth()          step()
                        (stages 1-3 SKIPPED)         (full pipeline)

Object detection:       Passed as arg               YOLO + SequenceToSeq
                        (predicted_object)           ClassAPI extraction

Motion command goes to: env.step(ActionXY/Rot)       sport_client.Move()
                        (updates simulation)         (moves real robot)

Image source:           None (no camera in sim)      Go2/RealSense/webcam

LiDAR source:           None                         Unitree UDP topic

Threading:              Single-threaded loop          5 threads
                                                     (image, YOLO obj,
                                                      YOLO pose, motion,
                                                      LiDAR)
```
