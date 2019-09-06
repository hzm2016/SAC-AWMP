import roboschool, gym
import datetime
import cv2
from TD3 import TD3
# from PIL import Image
from OpenGL import GLU
# from gym import wrappers

# env_name = "RoboschoolWalker2d-v1"
# env = gym.make(env_name)
# state = env.reset()
# action = env.action_space.sample()
# state, reward, done, _ = env.step(action)
# env.render()

def test():
    env_name = "RoboschoolWalker2d-v1"
    random_seed = 0
    n_episodes = 1
    lr = 0.001
    max_timesteps = 2000
    save_video = False

    filename = "TD3_{}_{}".format(env_name, random_seed)
    filename += ''
    result_path = '../../results'
    directory = result_path + "/models/TD3/{}".format(env_name)  # save trained models
    # directory = "./preTrained/{}".format(env_name)

    env = gym.make(env_name)
    # env = wrappers.Monitor(env, '../../results/video/{}_TD3_{}'.format(
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    #     env_name))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)

    policy.load_actor(directory, filename)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = '../../results/video/{}_TD3_{}.mp4'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        env_name)
    out_video = cv2.VideoWriter(video_name, fourcc, 60.0, (600, 400))
    print(video_name)

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if save_video:
                 img = env.render(mode = 'rgb_array')
                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
    
    