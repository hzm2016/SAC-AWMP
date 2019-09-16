import numpy as np
import torch
import roboschool, gym
import argparse
import os
import datetime
import TD3
import ATD3
import cv2
import glob
import sys
sys.path.insert(0,'../')
from utils import utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy import signal


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    # print ("---------------------------------------"                      )
    # print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print ("---------------------------------------"                      )
    return avg_reward


def main(method_name = '', policy_name = 'TD3', env_name = "RoboschoolWalker2d-v1",
         state_noise = 0.0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default=policy_name)  # Policy name
    parser.add_argument("--env_name", default=env_name)  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/ATD3_all')
    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--method_name", default=method_name,
                        help='Name of your method (default: )')  # Name of the method

    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=state_noise, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    # file_name = "%s_%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed), args.method_name)
    file_name = "TD3_%s_%s_%s" % (args.env_name, str(args.seed), args.method_name)
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    result_path = "../../results"
    video_dir = '{}/video/{}_{}'.format(result_path, args.env_name, args.method_name)
    model_dir = '{}/models/TD3/{}_{}'.format(result_path, args.env_name, args.method_name)
    if args.save_models and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.save_video and not os.path.exists(video_dir):
        os.makedirs(video_dir)


    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if 'TD3' == args.policy_name:
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif 'ATD3' == args.policy_name:
        policy = ATD3.ATD3(state_dim, action_dim, max_action)

    if not args.eval_only:

        log_dir = '{}/{}/{}_{}_{}_{}'.format(result_path, args.log_path,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                args.policy_name, args.env_name, args.method_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        replay_buffer = utils.ReplayBuffer()

        # Evaluate untrained policy
        evaluations = [evaluate_policy(env, policy)]

        total_timesteps = 0
        pre_num_steps = total_timesteps

        timesteps_since_eval = 0
        episode_num = 0
        done = True
        pbar = tqdm(total=args.max_timesteps, initial=total_timesteps, position=0, leave=True)
        best_reward = 0.0

        # TesnorboardX
        writer = SummaryWriter(logdir=log_dir)

        while total_timesteps < args.max_timesteps:
            if done:
                pbar.update(total_timesteps - pre_num_steps)
                pre_num_steps = total_timesteps
                if total_timesteps != 0:
                    writer.add_scalar('ave_reward/train', episode_reward, total_timesteps)
                    # print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
                    # 	  (total_timesteps, episode_num, episode_timesteps, episode_reward))
                    if args.policy_name == "TD3":
                        policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                     args.policy_noise, args.noise_clip, args.policy_freq)
                    else:
                        policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    avg_reward = evaluate_policy(env, policy)
                    evaluations.append(avg_reward)
                    writer.add_scalar('ave_reward/test', avg_reward, total_timesteps)
                    if best_reward < avg_reward:
                        best_reward = avg_reward
                        print(("Best reward! Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
                              (total_timesteps, episode_num, episode_timesteps, avg_reward))
                        if args.save_models:
                            policy.save(file_name, directory=model_dir)
                            policy.save(file_name, directory=log_dir)
                        np.save(log_dir + "/test_accuracy", evaluations)
                        utils.write_table(log_dir + "/test_accuracy", np.asarray(evaluations))
                    else:
                        print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
                              (total_timesteps, episode_num, episode_timesteps, avg_reward))

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1


            # Select action randomly or according to policy
            if total_timesteps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(obs))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward


            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

            # Store data in replay buffer
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # Final evaluation
        evaluations.append(evaluate_policy(env, policy))
        np.save(log_dir + "/test_accuracy", evaluations)
        utils.write_table(log_dir + "/test_accuracy", np.asarray(evaluations))
        env.close()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(10):
            model_path = result_path + '/runs/ATD3_all/TD3_{}_{}'.format(args.method_name, i+1)
            print(model_path)
            policy.load("%s" % (file_name), directory=model_path)
            if args.save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_name = video_dir + '/{}_{}_{}.mp4'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    file_name,args.state_noise)
                out_video = cv2.VideoWriter(video_name, fourcc, 60.0, (640, 480))
            for _ in range(1):
                obs = env.reset()
                obs_mat = np.asarray(obs)
                done = False
                reward_Q1_Q2_mat = np.zeros((0, 3))

                while not done:
                    action = policy.select_action(np.array(obs))

                    reward_Q1_Q2 = np.zeros((1,3))
                    Q1, Q2 = policy.critic(torch.FloatTensor(obs.reshape((1, -1))).to(device),
                                           torch.FloatTensor(action.reshape((1, -1))).to(device))
                    reward_Q1_Q2[0, 1] = Q1.cpu().detach().numpy()
                    reward_Q1_Q2[0, 2] = Q2.cpu().detach().numpy()

                    obs, reward, done, _ = env.step(action)

                    reward_Q1_Q2[0, 0] = reward
                    reward_Q1_Q2_mat = np.r_[reward_Q1_Q2_mat, reward_Q1_Q2]

                    obs[8:20] += np.random.normal(0, args.state_noise, size=obs[8:20].shape[0]).clip(
                                -1, 1)
                    obs_mat = np.c_[obs_mat, np.asarray(obs)]

                    if args.save_video:
                        img = env.render(mode='rgb_array')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_video.write(img)
                    else:
                        env.render()

            if args.save_video:
                utils.write_table(video_name + '_state', np.transpose(obs_mat))
                utils.write_table(video_name + '_reward_Q', reward_Q1_Q2_mat)
                out_video.release()
        env.close()


if __name__ == "__main__":
    env_name_vec = ['RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1',
                    'RoboschoolHumanoid-v1', 'RoboschoolAtlasForwardWalk-v1']
    policy_name_vec = ['TD3', 'ATD3']
    for r in range(4):
        for c in range(2):
            for n in range(10):
                print('r: {}, c: {}.'.format(r, c))
                main(policy_name = policy_name_vec[c],
                     env_name=env_name_vec[r])