# Deep RL Quadcopter Controller (Udacity Deep Learning Nano-degree final project)

Implemented deep deterministic policy gradient to control a quadcopter to finish certain task. (see: Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep Reinforcement Learning)

Files:

(1) actor.py: contains actor network, use double network. (see: Hasselt, H. v.; Guez, A.; Silver, D., 2015. Deep Reinforcement Learning with Double Q-learning) 

(2) critic.py: contains actor network, use double network and dueling network. (see: Wang, Z.; Schaul, T.; Hessel, M.;Hasselt, H. v.; Lanctot, M.; Freitas, N. d., 2016. Dueling Network Architectures for Deep Reinforcement Learning)

(3) replay.py: define memory replay (see: Mnih, V. et. al., 2015. Human-level control through deep reinforcement learning)

(4) ounoise: Ornstein–Uhlenbeck Noise, used for random action sampling.

(5) ddpg.py: combine all components to define deep deterministic policy gradient module.

(6) task.py: define task of the agent

(7) physics_sim.py: physical simulation of the agent

(8) main.py: training of the agent

Reference: https://github.com/songrotek/DDPG
