from ddpg import DDPG
from task import Task
import numpy as np
import sys

num_episodes = 1000
target_pos = np.array([5., 5., 5.])
init_pos = np.array([-5., -5., 0., 0., 0., 0.])
task = Task(init_pose=init_pos, target_pos=target_pos, runtime=20.)
agent = DDPG(task)
result = []
fout = open("reward.dat", 'w')

for i_episode in range(1, num_episodes+1):
    state = agent.reset() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent.score, agent.best_score), end="")  # [debug]
            result.append(agent.score)
            fout.write(str(i_episode) + '   ' +  str(agent.score) + '\n')
            break
    sys.stdout.flush()
    fout.flush()
