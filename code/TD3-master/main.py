import numpy as np
import torch
import roboschool, gym
import argparse
import os
import datetime
import utils
import TD3
import OurDDPG
import DDPG
import cv2

from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy import signal


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="RoboschoolWalker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--method_name", default="human_angle",
                        help='Name of your method (default: )')  # Name of the method
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3.1e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
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

    # if not os.path.exists("../../results"):
    #     os.makedirs("./results")
    # if args.save_models and not os.path.exists("./pytorch_models"):
    #     os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    if not args.eval_only:

        log_dir = '{}/runs/{}_TD3_{}_{}'.format(result_path,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                args.env_name, args.method_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        replay_buffer = utils.ReplayBuffer()

        # Evaluate untrained policy
        evaluations = [evaluate_policy(policy)]

        total_timesteps = 0
        pre_num_steps = total_timesteps

        if 'human_angle' == args.method_name:
            human_joint_angle = utils.read_table()

            pre_foot_contact = 1
            foot_contact = 1
            foot_contact_vec = np.asarray([1, 1, 1])
            delay_num = foot_contact_vec.shape[0] - 1
            gait_num = 0
            joint_angle = np.zeros((0, 7))
            idx_angle = np.zeros(0)
            reward_angle = np.zeros(0)

        timesteps_since_eval = 0
        episode_num = 0
        done = True
        pbar = tqdm(total=args.max_timesteps, initial=total_timesteps)
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
                    avg_reward = evaluate_policy(policy)
                    evaluations.append(avg_reward)
                    writer.add_scalar('ave_reward/test', avg_reward, total_timesteps)
                    if best_reward < avg_reward:
                        best_reward = avg_reward
                        print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
                              (total_timesteps, episode_num, episode_timesteps, avg_reward))
                        if args.save_models: policy.save(file_name, directory=model_dir)
                        np.save(log_dir + "/%s" % (file_name), evaluations)

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if 'human_angle' == args.method_name:
                    pre_foot_contact = 1
                    foot_contact = 1
                    foot_contact_vec = np.asarray([1, 1, 1])
                    gait_num = 0
                    joint_angle = np.zeros((0, 7))
                    idx_angle = np.zeros(0)
                    reward_angle = np.zeros(0)

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

            utils.fifo_list(foot_contact_vec, new_obs[-2])
            if 'human_angle' == args.method_name:
                if 0 == np.std(foot_contact_vec):
                    foot_contact = np.mean(foot_contact_vec)
                if 1 == (foot_contact - pre_foot_contact):
                    if joint_angle.shape[0] > 20:
                        gait_num += 1
                        if gait_num >= 2:
                            joint_angle_sampled = signal.resample(joint_angle[:-delay_num, :-1],
                                                                  num=human_joint_angle.shape[0])
                            coefficient = utils.calc_cos_similarity(human_joint_angle,
                                                                    joint_angle_sampled)
                            # print('gait_num:', gait_num, 'time steps in a gait: ', joint_angle.shape[0],
                            #       'coefficient: ', coefficient)
                            replay_buffer.add_final_reward(coefficient, joint_angle.shape[0] - delay_num,
                                                           delay= delay_num)
                            replay_buffer.add_specific_reward(reward_angle, idx_angle)
                            idx_angle = np.r_[idx_angle, joint_angle[:-delay_num, -1]]
                            reward_angle = np.r_[reward_angle,
                                                 0.05 * np.ones(joint_angle[:-delay_num, -1].shape[0])]
                    joint_angle = joint_angle[-delay_num:]
                pre_foot_contact = foot_contact
                joint_angle_obs = np.zeros((1, 7))
                joint_angle_obs[0, :-1] = obs[8:20:2]
                joint_angle_obs[-1] = total_timesteps
                joint_angle = np.r_[joint_angle, joint_angle_obs]

            reward -= 0.5

            if np.array_equal(new_obs[-2:], np.asarray([1., 1.])):
                still_steps += 1
            else:
                still_steps = 0
            if still_steps > 300:
                replay_buffer.add_final_reward(-0.5, still_steps - 1)
                reward -= 0.5
                done = True


            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)


            # Store data in replay buffer
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1



        # Final evaluation
        evaluations.append(evaluate_policy(policy))
        if args.save_models: policy.save("%s" % (file_name), directory=model_dir)
        np.save(log_dir + "/%s" % (file_name), evaluations)

    policy.load("%s" % (file_name), directory=model_dir)
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = video_dir + '/{}_TD3_{}.mp4'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_name)
        out_video = cv2.VideoWriter(video_name, fourcc, 60.0, (640, 480))
        print(video_name)
    for i in range(3):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            if args.save_video:
                img = env.render(mode='rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_video.write(img)
            else:
                env.render()
    env.close()
    if args.save_video:
        out_video.release()