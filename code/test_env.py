import pybullet_envs, roboschool, gym
import pybullet_envs.env_bases
import time
import pybullet
import cv2
pybullet.connect(pybullet.DIRECT)
env = gym.make("HopperBulletEnv-v0")
# env._render_width = 1280
# env._render_height = 720
# env.render(mode="human")
obs = env.reset()

for i in range(1000):
    print(i)
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    # env.render('rgb_array')
    img = env.render(mode='rgb_array')
    cv2.imshow('img', img)
    cv2.waitKey(1)
    # print(img.shape)