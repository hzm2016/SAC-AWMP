# import sys
# sys.path.remove('/home/kuangen/catkin_ws/devel/lib/python2.7/dist-packages')
import cv2
import roboschool, gym
import datetime
from TD3 import TD3
from PIL import Image
from OpenGL import GLU
from gym import wrappers



def test():
    env_name = "RoboschoolWalker2d-v1"
    random_seed = 0
    n_episodes = 3
    lr = 0.001
    max_timesteps = 2000
    save_video = True
    
    filename = "TD3_{}_{}".format(env_name, random_seed)
    filename += '_solved'
    directory = "./preTrained/{}".format(env_name)

    env = gym.make(env_name)
    # env = wrappers.Monitor(env, '../../results/video/{}_TD3_{}'.format(
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    #     env_name))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    
    policy.load_actor(directory, filename)


    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_name = '../../results/video/{}_TD3_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        env_name)
    out_video = cv2.VideoWriter(video_name, fourcc, 30.0, (600, 400))

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if save_video:
                 img = env.render(mode = 'rgb_array')
                 out_video.write(img)
                 # img = Image.fromarray(img)
                 # img.save('./gif/{}.jpg'.format(t))
            else:
                env.render()
            if done:
                break
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
        out_video.release()
                
if __name__ == '__main__':

    test()
    
    
    
