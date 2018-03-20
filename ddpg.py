import tensorflow as tf
import numpy as np
from actor import ActorNet
from critic import CriticNet
from replay import Replay
from ounoise import OUNoise

class DDPG:
    def __init__(self, task):
        # Hyper parameters
        self.learning_rate = 1e-5
        self.gamma = 0.99
        self.tau = 0.01

        # Define net
        self.sess = tf.Session()
        self.task = task
        self.learning_rate = learning_rate
        self.tau = tau
        self.actor = ActorNet(self.sess, self.task.state_size, self.task.action_size, self.learning_rate, \
                     self.task.action_low, self.task.action_high, self.tau)
        self.critic = CriticNet(self.sess, self.task.state_size, self.task.action_size, self.learning_rate, self.tau)

        # Define noise
        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2
        self.noise = OUNoise(self.task.action_size, self.mu, self.theta, self.sigma)

        # Define memory replay
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = Replay(self.buffer_size, self.batch_size)

    def reset(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def learn(self, experience):
        # Turn into different np arrays
        state_batch = np.vstack([e[0] for e in experience])
        action_batch = np.vstack([e[1] for e in experience])
        reward_batch = np.vstack([e[2] for e in experience])
        next_state_batch = np.vstack([e[3] for e in experience])
        done_batch = np.vstack([e[4] for e in experience])

        # Calculate next_state q value
        next_action_batch = self.actor.target_actions(next_state_batch)
        next_q_targets = self.critic.targetQ(next_state_batch, next_action_batch)

        # Train critic net
        q_targets = reward_batch + gamma * next_q_targets * (1 - done_batch)
        self.critic.train(state_batch, action_batch, q_targets)

        # Train actor net
        action_gradients = self.critic.gradients(state_batch, action_batch)
        self.actor.train(action_gradients, state_batch)

        # Update target network
        self.actor.update_target(False)
        self.critic.update_target(False)

    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)

        if len(self.memory.buffer) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        self.last_state = next_state

    def act(self, states):
        action = self.actor.actions(states)
        return action + self.noise.sample()





