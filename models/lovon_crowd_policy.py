"""
LOVON Policy for CrowdNav
==========================
Wraps the LOVON L2MM model and SocialNavigator as a CrowdNav-compatible Policy.

The CrowdNav environment provides ground-truth human states (px, py, vx, vy, radius)
rather than camera/LiDAR data. This policy translates those into the formats
expected by L2MM and SocialNavigator.

Usage:
    from models.lovon_crowd_policy import LOVONCrowdPolicy

    policy = LOVONCrowdPolicy()
    policy.configure(policy_config)           # standard CrowdNav config
    policy.load_lovon(model_path, tokenizer_path)  # LOVON-specific loading

    robot.set_policy(policy)
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

    # ------------------------------------------------------------------ #
    #  CrowdNav interface                                                  #
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

        # --- 4. Convert to CrowdNav action -------------------------------
        return self._to_action(motion_vector, self_state)

    # ------------------------------------------------------------------ #
    #  LOVON loading                                                       #
    # ------------------------------------------------------------------ #

    def load_lovon(self, model_path, tokenizer_path, social_nav_enabled=True):
        """
        Load the LOVON models. Call this after configure().

        Args:
            model_path      : path to L2MM model directory
            tokenizer_path  : path to tokenizer directory
            social_nav_enabled : whether to enable SocialNavigator
        """
        # --- L2MM ---------------------------------------------------------
        # from models.api_language2mostion import MotionPredictor
        # self.l2mm = MotionPredictor(
        #     model_path=model_path,
        #     tokenizer_path=tokenizer_path,
        # )

        # --- SocialNavigator ---------------------------------------------
        # from models.api_social_navigator import SocialNavigator
        # self.social_nav = SocialNavigator(enabled=social_nav_enabled)

        pass  # uncomment the above once paths are set

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
        Translate CrowdNav ground-truth state into the dict that
        MotionPredictor.predict() expects.

        This is where you bridge the two representations.
        CrowdNav gives metric world-frame positions; L2MM expects
        language + normalised object detections. Adapt as needed.
        """
        # Compute direction and distance to goal
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        goal_dist = np.hypot(dx, dy)
        goal_angle = np.arctan2(dy, dx) - self_state.theta

        # Placeholder: you'll need to decide how to map CrowdNav's
        # state into the fields L2MM was trained on. For example,
        # you could treat the goal as the "object" with a synthetic
        # bounding box, or you could bypass L2MM entirely and only
        # use SocialNavigator on top of a simpler goal-seeking policy.
        l2mm_input = {
            "mission_instruction_0": self.mission_instruction_0,
            "mission_instruction_1": self.mission_instruction_1,
            "predicted_object": self.predicted_object,
            "confidence": [1.0],
            "object_xyn": [np.clip(goal_angle / np.pi, -1, 1), 0.0],
            "object_whn": [0.1, 0.1],
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
        Run the SocialNavigator's step() using ground-truth human positions
        from CrowdNav, bypassing the perception/LiDAR pipeline.
        """
        if self.social_nav is None:
            return motion_vector

        # Build a synthetic pose_state dict from ground-truth.
        # SocialNavigator._parse_pose_state expects:
        #   {"num_people": int, "poses": [...], "pose_boxes": [...]}
        #
        # Since we have ground-truth metric positions, a cleaner path is
        # to directly populate SocialNavigator.tracked_humans and skip
        # the perception stages. That's a one-time modification to
        # SocialNavigator -- for now, pass the raw data through:
        pose_state = self._humans_to_pose_state(self_state, human_states)

        motion_vector = self.social_nav.step(
            motion_vector=motion_vector,
            pose_state=pose_state,
            mission_state=self.mission_state_in,
            lidar_ranges=None,
        )
        return motion_vector

    def _humans_to_pose_state(self, self_state, human_states):
        """
        Convert CrowdNav human ObservableStates into the pose_state dict
        that SocialNavigator expects.

        This is the main adaptation point. CrowdNav gives world-frame
        (px, py, vx, vy, radius); SocialNavigator expects pixel-space
        bounding boxes and optional keypoints.

        Options:
          A) Synthesise fake bounding boxes from world positions (quick hack).
          B) Directly inject TrackedHuman objects into social_nav, bypassing
             the perception pipeline (cleaner, recommended long-term).
        """
        # Placeholder -- return empty so SocialNavigator sees no people
        # and acts as passthrough. Fill this in with option A or B.
        return {"num_people": 0, "poses": [], "pose_boxes": []}

    def _to_action(self, motion_vector, self_state):
        """
        Convert L2MM's [vx, vy, omega_z] into a CrowdNav action.
        """
        vx, vy, omega_z = motion_vector

        if self.kinematics == "holonomic":
            return ActionXY(vx=vx, vy=vy)
        else:
            # unicycle: (speed, rotation)
            speed = np.hypot(vx, vy)
            speed = min(speed, self_state.v_pref)
            return ActionRot(v=speed, r=omega_z)
