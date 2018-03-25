import numpy as np
from physics_sim import PhysicsSim

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
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.len = np.linalg.norm(self.target_pos - init_pose[:3])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = max(1.0 - np.linalg.norm(self.target_pos - self.sim.pose[:3])/self.len, -1)
        return reward

    def self_defined_reward(self):
        die = False
        # Only get reward if get to visinity of target
        reward = max(-1, 1.-.1*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        z = self.mountainHeight(self.sim.pose[0], self.sim.pose[1])
        # Crash into mountain, get reward of -1
        if self.sim.pose[2] - z < -1e-5:
            reward -= 1
            die = True
        return reward, die

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
#            next_reward, die = self.self_defined_reward()
#            reward += next_reward
#            if die: done = True
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    def mountainHeight(self, x, y):
        z = max(0, 10 * np.exp(-(x - 5.) ** 2 / 10.0 - (y - 5.) ** 2 / 10.0) - 1.)
        return z
