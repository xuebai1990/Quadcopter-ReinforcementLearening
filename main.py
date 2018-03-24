from ddpg import DDPG
from task import Task
import numpy as np
import sys
import csv

num_episodes = 1000
target_pos = np.array([0., 10., 10.])
init_pos = np.array([0., 0., 0., 0., 0., 0.])
task = Task(init_pose=init_pos, target_pos=target_pos, runtime=5.)
agent = DDPG(task)
result = []
fout = open("reward.dat", 'w')
fout2 = open("physical_info.csv", 'w')

labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x : [] for x in labels}
writer = csv.writer(fout2)
writer.writerow(labels)

for i_episode in range(1, num_episodes+1):
    state = agent.reset() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

        # Write info to file
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        writer.writerow(to_write)

        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent.score, agent.best_score), end="")  # [debug]
            result.append(agent.score)
            fout.write(str(i_episode) + '   ' +  str(agent.score) + '\n')
            break
    sys.stdout.flush()
    fout.flush()
