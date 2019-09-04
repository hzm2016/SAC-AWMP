import torch
import roboschool, gym
import numpy as np
import datetime
from TD3 import TD3
from utils import ReplayBuffer
from tqdm import trange
from OpenGL import GLU
from tensorboardX import SummaryWriter
from walker2d_utils import *

def train():
    ######### Hyperparameters #########
    env_name = "RoboschoolWalker2d-v1"
    method_name = 'impedance'
    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 2000         # max num of episodes
    max_timesteps = 2000        # max timesteps in one episode
    result_path = '../../results'
    directory = result_path + "/models/TD3/{}_{}".format(env_name, method_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)

    time_step = 0.0165 / 4.0
    ###################################
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # action_dim = 18
    max_action = float(env.action_space.high[0])
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ave_dist = 0.0
    best_avg_reward = 0.0
    ep_reward = 0
    log_f = open("log.txt","w+")

    # TesnorboardX
    writer = SummaryWriter(
        logdir=result_path + '/runs/{}_TD3_{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            env_name))

    # training procedure:
    for episode in trange(1, max_episodes+1):
        state = env.reset()

        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=action_dim)

            torque, _ = calc_torque(state, action)

            torque = torque.clip(env.action_space.low, env.action_space.high)

            v_xyz = state[3:6]
            ave_dist += v_xyz[0] * time_step
            # take action in env:
            next_state, reward, done, _ = env.step(torque)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward

            ep_reward += reward
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            writer.add_scalar('ave_reward/train', avg_reward, episode)
            ave_dist = ave_dist / log_interval
            writer.add_scalar('ave_dist/train', ave_dist, episode)
            if avg_reward > best_avg_reward:
                policy.save(directory, filename)
                print("Episode: {}\tBest average Reward: {}\t Average distance: {}".format(
                    episode, avg_reward, ave_dist))

                best_avg_reward = avg_reward

            # if avg reward > 1300 then save and stop traning:
            if avg_reward >= 1500:
                print("########## Solved! ###########")
                name = filename + '_solved'
                policy.save(directory, name)
                log_f.close()
                break
            # print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0


if __name__ == '__main__':
    train()
    
