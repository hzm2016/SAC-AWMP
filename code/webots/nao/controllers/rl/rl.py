import os
import sys
project_path = '../../../../../'
sys.path.insert(0, project_path + 'code')
sys.path.insert(0, project_path + 'code/TD3')
print(os.getcwd())
print(sys.path)
import numpy as np
import torch
import argparse
import datetime
import TD3, ATD3, ATD3_CNN, ATD3_RNN, TD3_RNN
import cv2
import glob
from utils import utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy import signal
from gym_webots import Nao

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, args, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        if 'seq' in args.method_name:
            obs_vec = np.dot(np.ones((args.seq_len, 1)), obs.reshape((1, -1)))
        done = False
        while not done:
            if 'seq' in args.method_name:
                action = policy.select_action(np.array(obs_vec))
            else:
                action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            if 'seq' in args.method_name:
                obs_vec = utils.fifo_data(obs_vec, obs)
                # print('obs_vec: ', obs_vec)
            avg_reward += reward

    avg_reward /= eval_episodes
    # print ("---------------------------------------"                      )
    # print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print ("---------------------------------------"                      )
    return avg_reward


def main(method_name = '', policy_name = 'TD3', state_noise = 0.0, seed = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default=policy_name)  # Policy name
    parser.add_argument("--env_name", default="Webots_Nao")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/ATD3_walker2d')

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--method_name", default=method_name,
                        help='Name of your method (default: )')  # Name of the method

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved

    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=state_noise, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    args.seed += args.ini_seed
    args.seed = args.seed % 10
    file_name = "TD3_%s_%s_%s" % (args.env_name, args.seed, args.method_name)

    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")


    result_path = project_path + "results"
    video_dir = '{}/video/{}_{}'.format(result_path, args.env_name, args.method_name)
    model_dir = '{}/models/TD3/{}_{}'.format(result_path, args.env_name, args.method_name)
    if args.save_models and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.save_video and not os.path.exists(video_dir):
        os.makedirs(video_dir)

    env = Nao(action_dim=6, obs_dim=22)
    # env = gym.make(args.env_name)

    # Set seeds
    # env.seed(args.seed)

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
    elif 'ATD3_CNN' == args.policy_name:
        policy = ATD3_CNN.ATD3_CNN(state_dim, action_dim, max_action, args.seq_len)
    elif 'ATD3_RNN' == args.policy_name:
        policy = ATD3_RNN.ATD3_RNN(state_dim, action_dim, max_action)
    elif 'TD3_RNN' == args.policy_name:
        policy = TD3_RNN.TD3_RNN(state_dim, action_dim, max_action)

    if not args.eval_only:

        log_dir = '{}/{}/seed_{}_{}_{}_{}_{}'.format(result_path, args.log_path, args.seed,
                                                datetime.datetime.now().strftime("%d_%H-%M-%S"),
                                                args.policy_name, args.env_name, args.method_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        replay_buffer = utils.ReplayBuffer()

        # Evaluate untrained policy
        evaluations = [evaluate_policy(env, policy, args)]

        total_timesteps = 0
        pre_num_steps = total_timesteps

        if 'human_angle' in args.method_name:
            human_joint_angle = utils.read_table(file_name= project_path + 'data/joint_angle.xls')
            pre_foot_contact = 1
            foot_contact = 1
            foot_contact_vec = np.asarray([1, 1, 1])
            delay_num = foot_contact_vec.shape[0] - 1
            gait_num = 0
            joint_angle = np.zeros((0, 7))
            idx_angle = np.zeros(0)
            reward_angle = np.zeros(0)

        if 'still_steps' in args.method_name:
            still_steps = 0

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
                    avg_reward = evaluate_policy(env, policy, args)
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
                    # else:
                        # print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
                        #       (total_timesteps, episode_num, episode_timesteps, avg_reward))

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if 'seq' in args.method_name:
                    obs_vec = np.dot(np.ones((args.seq_len, 1)), obs.reshape((1, -1)))

                if 'human_angle' in args.method_name:
                    pre_foot_contact = 1
                    foot_contact = 1
                    foot_contact_vec = np.asarray([1, 1, 1])
                    gait_num = 0
                    joint_angle = np.zeros((0, 7))
                    idx_angle = np.zeros(0)
                    reward_angle = np.zeros(0)
                if 'still_steps' in args.method_name:
                    still_steps = 0

            # Select action randomly or according to policy
            if total_timesteps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                if 'seq' in args.method_name:
                    action = policy.select_action(np.array(obs_vec))
                else:
                    action = policy.select_action(np.array(obs))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if 'human_angle' in args.method_name:
                foot_contact_vec = utils.fifo_data(foot_contact_vec, new_obs[-2])
                if 0 == np.std(foot_contact_vec):
                    foot_contact = np.mean(foot_contact_vec)
                if 1 == (foot_contact - pre_foot_contact):
                    if joint_angle.shape[0] > 20:
                        gait_num += 1
                        if gait_num >= 2:
                            joint_angle_sampled = signal.resample(joint_angle[:-delay_num, :-1],
                                                                  num=human_joint_angle.shape[0])
                            # The ATD3 seems to prefer the negative similarity reward
                            coefficient = utils.calc_cos_similarity(human_joint_angle,
                                                                    joint_angle_sampled) - 0.5
                            # print('gait_num:', gait_num, 'time steps in a gait: ', joint_angle.shape[0],
                            #       'coefficient: ', coefficient)
                            replay_buffer.add_final_reward(coefficient, joint_angle.shape[0] - delay_num,
                                                           delay=delay_num)
                            replay_buffer.add_specific_reward(reward_angle, idx_angle)
                            idx_angle = np.r_[idx_angle, joint_angle[:-delay_num, -1]]
                            reward_angle = np.r_[reward_angle,
                                                 0.05 * np.ones(joint_angle[:-delay_num, -1].shape[0])]
                    joint_angle = joint_angle[-delay_num:]
                pre_foot_contact = foot_contact
                joint_angle_obs = np.zeros((1, 7))
                joint_angle_obs[0, :-1] = obs[8:20:2]
                joint_angle_obs[0, -1] = total_timesteps
                joint_angle = np.r_[joint_angle, joint_angle_obs]
                reward -= 0.5

            if 'still_steps' in args.method_name:
                if np.array_equal(new_obs[-2:], np.asarray([1., 1.])):
                    still_steps += 1
                else:
                    still_steps = 0
                if still_steps > 100:
                    replay_buffer.add_final_reward(-2.0, still_steps - 1)
                    reward -= 2.0
                    done = True

            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

            if 'seq' in args.method_name:
                # Store data in replay buffer
                new_obs_vec = utils.fifo_data(np.copy(obs_vec), new_obs)
                replay_buffer.add((np.copy(obs_vec), new_obs_vec, action, reward, done_bool))
                # print('train obs_vec: ', obs_vec)
                # print('train new_obs_vec: ', new_obs_vec)
                # print('1 obs_vec: ', replay_buffer.get(-1))
                obs_vec = utils.fifo_data(obs_vec, new_obs)
                # print('2 obs_vec: ', replay_buffer.get(-1))
            else:
                replay_buffer.add((obs, new_obs, action, reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # Final evaluation
        evaluations.append(evaluate_policy(env, policy, args))
        np.save(log_dir + "/test_accuracy", evaluations)
        utils.write_table(log_dir + "/test_accuracy", np.asarray(evaluations))
        env.reset()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # for i in range(10):
        for i in [0]:
            model_path = result_path + '/runs/ATD3_walker2d/{}_{}'.format(args.method_name, i+1)
            print(model_path)
            policy.load("%s" % (file_name), directory=model_path)
            for _ in range(1):
                if args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_name = video_dir + '/{}_{}_{}.mp4'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        file_name, args.state_noise)
                    out_video = cv2.VideoWriter(video_name, fourcc, 60.0, (640, 480))
                obs = env.reset()
                if 'seq' in args.method_name:
                    obs_vec = np.dot(np.ones((args.seq_len, 1)), obs.reshape((1, -1)))

                obs_mat = np.asarray(obs)
                done = False
                reward_Q1_Q2_mat = np.zeros((0, 3))

                if 'human_angle' in args.method_name and args.save_video:
                    human_joint_angle = utils.read_table()

                    pre_foot_contact = 1
                    foot_contact = 1
                    foot_contact_vec = np.asarray([1, 1, 1])
                    delay_num = foot_contact_vec.shape[0] - 1
                    gait_num = 0
                    joint_angle = np.zeros((0, 7))
                    idx_angle = np.zeros(0, dtype=np.int)
                    reward_angle = np.zeros(0)

                if 'still_steps' in args.method_name and args.save_video:
                    still_steps = 0

                while not done:
                    if 'seq' in args.method_name:
                        action = policy.select_action(np.array(obs_vec))
                    else:
                        action = policy.select_action(np.array(obs))

                    reward_Q1_Q2 = np.zeros((1,3))
                    Q1, Q2 = policy.critic(torch.FloatTensor(np.expand_dims(obs_vec, axis=0)).to(device),
                                           torch.FloatTensor(np.expand_dims(action, axis=0)).to(device))
                    reward_Q1_Q2[0, 1] = Q1.cpu().detach().numpy()
                    reward_Q1_Q2[0, 2] = Q2.cpu().detach().numpy()

                    obs, reward, done, _ = env.step(action)

                    if 'seq' in args.method_name:
                        obs_vec = utils.fifo_data(obs_vec, obs)

                    if 'human_angle' in args.method_name and args.save_video:
                        foot_contact_vec = utils.fifo_data(foot_contact_vec, obs[-2])
                        if 0 == np.std(foot_contact_vec):
                            foot_contact = np.mean(foot_contact_vec)
                        if 1 == (foot_contact - pre_foot_contact):
                            if joint_angle.shape[0] > 20:
                                gait_num += 1
                                if gait_num >= 2:
                                    joint_angle_sampled = signal.resample(joint_angle[:-delay_num, :-1],
                                                                          num=human_joint_angle.shape[0])
                                    coefficient = utils.calc_cos_similarity(human_joint_angle,
                                                                            joint_angle_sampled) - 0.5
                                    # print('gait_num:', gait_num, 'time steps in a gait: ', joint_angle.shape[0],
                                    #       'coefficient: ', coefficient)
                                    reward_Q1_Q2_mat[-joint_angle.shape[0]: -delay_num, 0] += coefficient
                                    if len(idx_angle) > 0:
                                        reward_Q1_Q2_mat[idx_angle.astype(np.int)] += reward_angle.reshape((-1, 1))

                                    idx_angle = np.r_[idx_angle, joint_angle[:-delay_num, -1]]
                                    reward_angle = np.r_[reward_angle,
                                                         0.05 * np.ones(joint_angle[:-delay_num, -1].shape[0])]
                            joint_angle = joint_angle[-delay_num:]
                        pre_foot_contact = foot_contact
                        joint_angle_obs = np.zeros((1, 7))
                        joint_angle_obs[0, :-1] = obs[8:20:2]
                        joint_angle_obs[0, -1] = reward_Q1_Q2_mat.shape[0]
                        # print(joint_angle_obs)
                        joint_angle = np.r_[joint_angle, joint_angle_obs]
                        reward -= 0.5

                    if 'still_steps' in args.method_name and args.save_video:
                        if np.array_equal(obs[-2:], np.asarray([1., 1.])):
                            still_steps += 1
                        else:
                            still_steps = 0
                        if still_steps > 100:
                            reward_Q1_Q2_mat[(-still_steps + 1):, 0] += -2.0
                            reward -= 2.0
                            done = True

                    reward_Q1_Q2[0, 0] = reward
                    reward_Q1_Q2_mat = np.r_[reward_Q1_Q2_mat, reward_Q1_Q2]

                    obs[8:20] += np.random.normal(0, args.state_noise, size=obs[8:20].shape[0]).clip(
                                -1, 1)
                    obs_mat = np.c_[obs_mat, np.asarray(obs)]
                if args.save_video:
                    utils.write_table(video_name + '_state', np.transpose(obs_mat))
                    utils.write_table(video_name + '_reward_Q', reward_Q1_Q2_mat)
                    out_video.release()
        env.reset()

if __name__ == "__main__":
    method_name_vec = ['human_angle_still_steps_seq_ATD3_RNN', 'human_angle_still_steps_ATD3',
                       'human_angle_still_steps', 'still_steps', '']
    policy_name_vec = ['ATD3_RNN', 'ATD3', 'TD3', 'TD3', 'TD3']
    for r in range(1):
        for c in range(2):
            for n in range(1):
                print('r: {}, c: {}.'.format(r, c))
                main(method_name=method_name_vec[r], policy_name = policy_name_vec[r],
                     state_noise= 0.04 * n, seed=c)
