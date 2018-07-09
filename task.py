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

        self.state_size = self.action_repeat * 12
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
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = -min(abs(self.sim.pose[2] - self.target_pos[2]), 20.0)
        #if self.sim.pose[2] >= self.target_pos[2]:
        #    reward += 10.0
        #reward = 1 - 0.3*abs(self.sim.pose[:3] - self.target_pos).sum() + 0.2 * min(self.sim.v[2], 10) + 0.1 * max(self.sim.pose[2], 10)
        remaining_distance, vertical_velocity, height = self.get_flight_parameters()
        #take_off_reward = self.sim.time*max(height, 10) if self.sim.time < 0.8 else 0
        take_off_reward = -min(abs(self.sim.pose[2] - self.target_pos[2]), 10) + min(vertical_velocity, 1)

        #reward = 1 - 3*remaining_distance + take_off_reward

        reward = take_off_reward - remaining_distance

        if height < 0.01:
            reward -= 5

        # penalize crash
        #if done and self.sim.time < self.sim.runtime:
        #    reward -= 10

        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done)
            pose_all.extend([self.sim.pose, self.sim.v, self.sim.angular_v])
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v] * self.action_repeat)
        return state
