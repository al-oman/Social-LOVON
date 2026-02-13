"""
LOVON Policy for CrowdNav
==========================
"""

import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot


class LOVONCrowdPolicy(Policy):
    """
    CrowdNav Policy that delegates to LOVON's L2MM + SocialNavigator.

    Receives a JointState (robot FullState + list of human ObservableStates)
    and returns an ActionXY or ActionRot.
    """

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = "unicycle"  # or "holonomic" -- set in configure()
        self.name = "lovon"

        # LOVON components (loaded later)
        self.l2mm = None               # MotionPredictor instance
        self.social_nav = None         # SocialNavigator instance

        # Mission context -- set these before running episodes
        self.mission_instruction_0 = ""   # e.g. "go to the red chair"
        self.mission_instruction_1 = ""   # e.g. secondary instruction
        self.predicted_object = "none"
        self.mission_state_in = "running"
        self.search_state_in = "had_searching_0"

        # Camera parameters
        self.image_width = 640
        self.fov = 120

        self.goal_radius = 0.1

    # ------------------------------------------------------------------ #
    #  CrowdNav interfacing                                                  #
    # ------------------------------------------------------------------ #

    def configure(self, config):
        """
        Called by CrowdNav with a RawConfigParser from policy.config.
        Read any LOVON-specific settings from a [lovon] section, and
        fall back to standard sections for kinematics, etc.
        """
        if config.has_option("lovon", "kinematics"):
            self.kinematics = config.get("lovon", "kinematics")
        elif config.has_option("action_space", "kinematics"):
            self.kinematics = config.get("action_space", "kinematics")

        # Add any other config reads you need here.

    def predict(self, state):
        """
        CrowdNav calls this with a JointState:
            state.self_state  : FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta)
            state.human_states: [ObservableState(px, py, vx, vy, radius), ...]

        Returns:
            ActionXY(vx, vy)  if self.kinematics == "holonomic"
            ActionRot(v, r)   if self.kinematics == "unicycle"
        """
        self_state = state.self_state
        human_states = state.human_states

        # --- 1. Build L2MM input dict from CrowdNav state ----------------
        l2mm_input = self._build_l2mm_input(self_state, human_states)

        # --- 2. Get motion vector from L2MM ------------------------------
        l2mm_output = self._call_l2mm(l2mm_input)
        motion_vector = l2mm_output["motion_vector"]   # [vx, vy, omega_z]
        self.mission_state_in = l2mm_output["predicted_state"]
        self.search_state_in = l2mm_output["search_state"]

        # --- 3. (Optional) Run SocialNavigator to modulate velocity ------
        motion_vector = self._call_social_nav(
            motion_vector, self_state, human_states
        )

        # --- debug: inspect L2MM behavior near goal ---
        xyn, whn = self._angle_to_xyn_whn(self_state)
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        goal_dist = np.hypot(dx, dy)
        print(f"robot=({self_state.px:.2f},{self_state.py:.2f}) theta={self_state.theta:.2f} "
              f"dist={goal_dist:.2f} xyn={xyn[0]:.3f} whn={whn[0]:.2f} "
              f"state={self.mission_state_in} mv=[{motion_vector[0]:.3f},{motion_vector[1]:.3f},{motion_vector[2]:.3f}]")

        # --- 4. Convert to CrowdNav action -------------------------------
        return self._to_action(motion_vector, self_state)

    # ------------------------------------------------------------------ #
    #  LOVON loading                                                       #
    # ------------------------------------------------------------------ #

    def load_lovon(self, model_path, tokenizer_path, social_nav_enabled=True):
        """
        Load the LOVON models. Call this after configure().
        """
        # --- L2MM ---------------------------------------------------------
        from models.api_language2mostion import MotionPredictor
        self.l2mm = MotionPredictor(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

        from models.api_social_navigator import SocialNavigator
        self.social_nav = SocialNavigator(enabled=social_nav_enabled)

    def set_mission(self, instruction_0, instruction_1="", predicted_object="none"):
        """Set the language instruction for the current episode."""
        self.mission_instruction_0 = instruction_0
        self.mission_instruction_1 = instruction_1
        self.predicted_object = predicted_object
        # Reset state machine at episode start
        self.mission_state_in = "running"
        self.search_state_in = "had_searching_0"

    # ------------------------------------------------------------------ #
    #  Translation helpers                                                 #
    # ------------------------------------------------------------------ #

    def _build_l2mm_input(self, self_state, human_states):
        """
        takes CrowdNav states and creates input for L2MM
        """

        xyn, whn = self._angle_to_xyn_whn(self_state)

        l2mm_input = {
            "mission_instruction_0": self.mission_instruction_0,
            "mission_instruction_1": self.mission_instruction_1,
            "predicted_object": self.predicted_object,
            "confidence": [1.0],
            "object_xyn": xyn,
            "object_whn": whn,
            "mission_state_in": self.mission_state_in,
            "search_state_in": self.search_state_in,
        }
        return l2mm_input

    def _call_l2mm(self, l2mm_input):
        """Call L2MM or return a fallback if not loaded."""
        if self.l2mm is not None:
            return self.l2mm.predict(l2mm_input)

        # ---- Fallback: simple goal-seeking (no model) ----
        # Remove this once L2MM is loaded.
        return {
            "motion_vector": [0.0, 0.0, 0.0],
            "predicted_state": self.mission_state_in,
            "search_state": self.search_state_in,
        }

    def _call_social_nav(self, motion_vector, self_state, human_states):
        """
        Run the SocialNavigator using ground-truth human positions from CrowdNav.
        Bypasses perception stages 1-3 by calling step_ground_truth().
        """
        if self.social_nav is None:
            return motion_vector

        gt_humans = self._sim_humans_to_robot_frame(self_state, human_states)

        motion_vector = self.social_nav.step_ground_truth(
            motion_vector=motion_vector,
            gt_humans=gt_humans,
            mission_state=self.mission_state_in,
        )
        return motion_vector

    def _sim_humans_to_robot_frame(self, self_state, human_states):
        """
        Transform CrowdNav world-frame human states into robot-frame dicts
        for SocialNavigator.step_ground_truth().

        For each human:
          - Transform (px, py) to robot-frame [x_lateral, depth]
          - Transform (vx, vy) to robot-frame velocity

        Args:
            self_state:   robot FullState (px, py, theta, ...)
            human_states: list of ObservableState (px, py, vx, vy, radius)

        Returns:
            list of dicts: {track_id, position_rf, distance, velocity, radius}
        """
        gt_humans = []
        for i, h in enumerate(human_states):
            x_lat, depth = self.world_to_robot_frame(
                h.px, h.py, self_state.px, self_state.py, self_state.theta
            )
            distance = np.hypot(h.px - self_state.px, h.py - self_state.py)
            vx_lat, v_depth = self.world_to_robot_frame_velocity(
                h.vx, h.vy, self_state.theta
            )
            gt_humans.append({
                "track_id": i,
                "position_rf": [x_lat, depth],
                "distance": float(distance),
                "velocity": [vx_lat, v_depth],
                "radius": h.radius,
            })
        return gt_humans

    def _angle_to_xyn_whn(self, self_state):
        # Compute direction and distance to goal
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        goal_dist = np.hypot(dx, dy)
        goal_angle = np.arctan2(dy, dx) - self_state.theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        # clamp to FOV so pinhole model stays valid
        # (goal behind/beside robot gets pushed to image edge)
        half_fov = np.radians(self.fov / 2)
        goal_angle = np.clip(goal_angle, -half_fov, half_fov)

        # pinhole model of camera for angle to pixel
        # whn is normalized size of bounding box
        # xyn is normalized coords of centre of bounding box: [0.5, 0.5]

        x = 0.5 - np.tan(goal_angle) / (2 * np.tan(half_fov))
        xyn = [np.clip(x, 0.0, 1.0), 0.5]

        # placeholder logic for bounding box size prediction from sim state
        if goal_dist < self.goal_radius:
            print('approaching goal!')
            whn = [0.4, 0.4]
        else:
            whn = [0.1, 0.1] 


        return xyn, whn

    def _to_action(self, motion_vector, self_state):
        """
        Convert L2MM's [vx, vy, omega_z] into a CrowdNav action.
        """
        vx, vy, omega_z = motion_vector

        if self.kinematics == "holonomic":
            return ActionXY(vx=vx, vy=vy)
        else:
            # unicycle: (speed, rotation)
            # L2MM outputs omega_z as rad/s, but ActionRot.r is angle-per-step
            speed = np.hypot(vx, vy)
            speed = min(speed, self_state.v_pref)
            r = omega_z * self.time_step
            return ActionRot(v=speed, r=r)

    # ------------------------------------------------------------------
    #  Coordinate transform: world frame -> robot frame
    # ------------------------------------------------------------------

    def world_to_robot_frame(self, human_px, human_py, robot_px, robot_py, robot_theta):
        """
        Transform a world-frame position into robot-frame [x_lateral, depth].

        Robot frame convention (matching SocialNavigator):
            x_lateral: positive = right of robot
            depth:     positive = forward from robot
        """
        dx = human_px - robot_px
        dy = human_py - robot_py
        cos_t = np.cos(-robot_theta)
        sin_t = np.sin(-robot_theta)
        x_rot = dx * cos_t - dy * sin_t
        y_rot = dx * sin_t + dy * cos_t
        depth = x_rot
        x_lateral = -y_rot
        return x_lateral, depth

    def world_to_robot_frame_velocity(self, human_vx, human_vy, robot_theta):
        """
        Transform a world-frame velocity into robot-frame [vx_lateral, v_depth].
        Same rotation as position but without translation.
        """
        cos_t = np.cos(-robot_theta)
        sin_t = np.sin(-robot_theta)
        x_rot = human_vx * cos_t - human_vy * sin_t
        y_rot = human_vx * sin_t + human_vy * cos_t
        v_depth = x_rot
        vx_lateral = -y_rot
        return vx_lateral, v_depth