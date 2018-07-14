import numpy as np
from physics_sim import PhysicsSim
from scipy.spatial import distance

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 1

        self.state_size = self.action_repeat * 9
        self.action_low = 1
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_flight_parameters(self):
        remaining_distance = distance.euclidean(self.sim.pose[:3], self.target_pos)
        vertical_velocity = self.sim.v[2]
        height = self.sim.pose[2]
        return remaining_distance, vertical_velocity, height

    def get_reward(self, done):
        """Uses current pose and velocity of sim to return reward."""
        current_pos = self.sim.pose[:3]
        # Euclidean coordinates
        x, y, z = current_pos
        # velocities
        v_x, v_y, v_z = self.sim.v

        remaining_distance = abs(current_pos - self.target_pos).sum()
        angular_velocity_penalty = abs(self.sim.angular_v).sum()
        velocity_penalty = abs(abs(current_pos - self.target_pos).sum() - abs(self.sim.v).sum())
        deviation_penalty = 0.005*(v_x**2 + v_y**2) + 0.05*x**2 + 0.05*y**2

        reward = 1 + 0.2*min(v_z, 5) - 0.8*remaining_distance - 0.9*deviation_penalty - 2*velocity_penalty - 3*angular_velocity_penalty

        if done:
            # penalize crash
            if self.sim.time < self.sim.runtime:
                reward -= 30
            # extra reward if the quadcopter is near the target position
            elif remaining_distance < 1.5:
                print('You did it!')
                reward += 30

        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done)
            pose_all.extend([self.sim.pose, self.sim.v])
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v] * self.action_repeat)
        return state
