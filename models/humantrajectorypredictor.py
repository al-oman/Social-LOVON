import numpy as np
from collections import defaultdict, deque
# ====================================================================== #
#  Trajectory Predictor (adapted from SocialNav-Map: arXiv:2511.12232)   #
# ====================================================================== #
class HumanTrajectoryPredictor:
    """
    Linear-extrapolation trajectory predictor.

    Maintains a sliding window of recent positions per tracked human,
    fits a degree-1 polynomial (constant-velocity) via least-squares,
    and extrapolates forward.

    Inputs/outputs are in robot-frame metric coordinates [x_lateral, depth].
    """

    def __init__(self, history_length=5, prediction_steps=10, prediction_interval=1, step_stride=1,
                 dt=0.25, speed_min=0.15, speed_max=1.0):
        """
        Args:
            history_length:      Number of past positions to retain per human.
            prediction_steps:    How many output points to produce (at max speed).
            prediction_interval: Only recompute predictions every N calls.
            step_stride:         Frame-count increment per output point.
                                 e.g. stride=4 at dt=0.25s → 1 point per second.
            dt:                  Seconds per frame (for converting speed to m/s).
            speed_min:           Speed (m/s) at or below which only 1 point is produced.
            speed_max:           Speed (m/s) at or above which all prediction_steps are used.
        """
        self.history_length = history_length
        self.prediction_steps = prediction_steps
        self.prediction_interval = prediction_interval
        self.step_stride = step_stride
        self.dt = dt
        self.speed_min = speed_min
        self.speed_max = speed_max

        self.agent_trajectories = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.predicted_trajectories = {}  # {agent_id: [[x,y], ...]}
        self.last_prediction_step = -1

    def reset(self):
        """Clear all history and predictions."""
        self.agent_trajectories.clear()
        self.predicted_trajectories.clear()
        self.last_prediction_step = -1

    def update_agent_position(self, agent_id, position, timestep):
        """
        Append a new position observation.

        Args:
            agent_id:  Persistent track ID (int).
            position:  [x_lateral, depth] in meters, robot frame.
            timestep:  Monotonic frame counter (int).
        """
        self.agent_trajectories[agent_id].append({
            'position': list(position),
            'timestep': timestep,
        })

    def _estimate_speed(self, trajectory):
        """Estimate speed (m/s) from recent trajectory history."""
        if len(trajectory) < 2:
            return 0.0
        p0 = np.array(trajectory[0]['position'])
        p1 = np.array(trajectory[-1]['position'])
        d_frames = trajectory[-1]['timestep'] - trajectory[0]['timestep']
        if d_frames <= 0:
            return 0.0
        # Convert from meters/frame to meters/second
        return float(np.linalg.norm(p1 - p0)) / (d_frames * self.dt)

    def _speed_scaled_steps(self, speed):
        """Return number of prediction points based on estimated speed.

        Linearly interpolates between 1 (at speed_min) and
        prediction_steps (at speed_max). Clamps at both ends.
        """
        if speed <= self.speed_min:
            return 1
        if speed >= self.speed_max:
            return self.prediction_steps
        t = (speed - self.speed_min) / (self.speed_max - self.speed_min)
        return max(1, int(round(1 + t * (self.prediction_steps - 1))))

    def predict_trajectory(self, agent_id):
        """Predict future positions for one agent. Returns list of [x, y].

        The number of output points scales with the agent's estimated speed:
        slow/stationary agents get fewer points, fast agents get up to
        prediction_steps points.
        """
        trajectory = self.agent_trajectories[agent_id]

        if len(trajectory) < 2:
            return []

        positions = [entry['position'] for entry in trajectory]
        timesteps = [entry['timestep'] for entry in trajectory]

        speed = self._estimate_speed(trajectory)
        n_steps = self._speed_scaled_steps(speed)
        # n_steps = self.prediction_steps

        # Stationary check — all positions identical
        unique = set(tuple(p) for p in positions)
        if len(unique) == 1:
            return [list(positions[-1])]

        # Linear fit (degree 1) on each axis independently
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        x_poly = np.polyfit(timesteps, x_coords, 1)
        y_poly = np.polyfit(timesteps, y_coords, 1)

        last_t = timesteps[-1]
        predicted = []
        for i in range(1, n_steps + 1):
            ft = last_t + i * self.step_stride
            predicted.append([
                float(np.polyval(x_poly, ft)),
                float(np.polyval(y_poly, ft)),
            ])
        return predicted

    def predict_all(self, current_timestep):
        """
        Predict trajectories for all tracked agents (throttled).

        Returns:
            dict  {agent_id: [[x, y], ...]}
        """
        if (current_timestep - self.last_prediction_step) < self.prediction_interval:
            return self.predicted_trajectories

        self.last_prediction_step = current_timestep
        self.predicted_trajectories = {}

        for agent_id in list(self.agent_trajectories.keys()):
            pred = self.predict_trajectory(agent_id)
            if pred:
                self.predicted_trajectories[agent_id] = pred

        return self.predicted_trajectories

    def prune_stale(self, active_ids):
        """Remove history for agents no longer tracked."""
        stale = [aid for aid in self.agent_trajectories if aid not in active_ids]
        for aid in stale:
            del self.agent_trajectories[aid]
            self.predicted_trajectories.pop(aid, None)


