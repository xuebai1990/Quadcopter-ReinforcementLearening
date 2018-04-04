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
        self.action_high = 5000
        self.action_size = 4
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
#        self.len = np.linalg.norm(self.target_pos - init_pose[:3])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#        reward = max(1.0 - np.linalg.norm(self.target_pos - self.sim.pose[:3])/self.len, -1)
        done = False
        reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        if self.sim.pose[2] >= self.target_pos[2]:  # agent has crossed the target height
            reward += 20.0  # bonus reward
            done = True
        elif self.sim.time > self.runtime or self.sim.pose[2] < 0:  # agent has run out of time
            reward -= 20.0  # extra penalty
            done = True
        return reward, done

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
        over = False
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            next_reward, over = self.get_reward()
            pose_all.append(self.sim.pose)
            reward += next_reward
            if over: done = True
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
