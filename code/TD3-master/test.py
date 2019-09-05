import numpy as np
from utils import *
import roboschool, gym

env = gym.make("RoboschoolWalker2d-v1")
env.seed(0)
obs = env.reset()
done = False
idx = 0
joint_angle_mat = read_table()
print(joint_angle_mat)
init_state = True
while True:
    joint_angle_e = joint_angle_mat[idx, :]
    joint_angle = obs[8:20:2]
    joint_speed = obs[9:20:2]
    # if init_state:
    k = 0.1
    b = 0.1
    joint_angle_e = np.asarray([1, 1, 1, 1, 0, 0])
    # if np.linalg.norm(joint_angle_e - joint_angle) < 0.2:
    # 	init_state = False
    # else:
    # 	idx += 1
    # 	k = 0.5
    # 	b = 0.1
    action = k * (joint_angle_e - joint_angle) - b * joint_speed
    action[3] = 0
    obs, reward, done, _ = env.step(action)
    env.render()
