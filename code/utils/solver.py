import numpy as np
import os
import datetime
import cv2
import torch
from utils import utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy import signal
from methods import TD3, ATD3, ATD3_CNN, ATD3_RNN, TD3_RNN


class Solver(object):
    def __init__(self, args, env, project_path):
        args.seed += args.ini_seed
        args.seed = args.seed % 10
        self.args = args
        self.env = env

        self.file_name = "TD3_%s_%s_%s" % (args.env_name, args.seed, args.method_name)
        print("---------------------------------------")
        print("Settings: %s" % self.file_name)
        print("---------------------------------------")

        self.project_path = project_path
        self.result_path = "../results"

        # Set seeds
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        # Initialize policy
        if 'ATD3' == args.policy_name:
            policy = ATD3.ATD3(state_dim, action_dim, max_action)
        elif 'ATD3_CNN' == args.policy_name:
            policy = ATD3_CNN.ATD3_CNN(state_dim, action_dim, max_action, args.seq_len)
        elif 'ATD3_RNN' == args.policy_name:
            policy = ATD3_RNN.ATD3_RNN(state_dim, action_dim, max_action)
        elif 'TD3_RNN' == args.policy_name:
            policy = TD3_RNN.TD3_RNN(state_dim, action_dim, max_action)
        else:
            policy = TD3.TD3(state_dim, action_dim, max_action)
        self.policy = policy

        self.log_dir = '{}/{}/seed_{}_{}_{}_{}_{}'.format(self.result_path, self.args.log_path, self.args.seed,
                                                          datetime.datetime.now().strftime("%d_%H-%M-%S"),
                                                          self.args.policy_name, self.args.env_name,
                                                          self.args.method_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.replay_buffer = utils.ReplayBuffer()

        # Evaluate untrained policy
        self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]
        self.total_timesteps = 0
        self.pre_num_steps = self.total_timesteps
        self.timesteps_since_eval = 0
        self.episode_progress = 0.0
        self.best_reward = 0.0
        # TesnorboardX
        self.writer = SummaryWriter(logdir=self.log_dir)
        self.env_timeStep = 4

    def train_once(self):
        self.pbar.update(self.total_timesteps - self.pre_num_steps)
        self.pre_num_steps = self.total_timesteps

        if len(self.replay_buffer.storage) > self.env.frame:
            self.replay_buffer.add_final_reward(self.episode_progress / 1000.0,
                                                self.env.frame)
        if self.total_timesteps != 0:
            self.writer.add_scalar('ave_reward/train', self.episode_reward, self.total_timesteps)
            self.policy.train(self.replay_buffer, self.episode_timesteps, self.args.batch_size, self.args.discount,
                              self.args.tau, self.args.policy_noise, self.args.noise_clip, self.args.policy_freq)

        # Evaluate episode
        if self.timesteps_since_eval >= self.args.eval_freq:
            self.timesteps_since_eval %= self.args.eval_freq
            avg_reward = evaluate_policy(self.env, self.policy, self.args)
            self.evaluations.append(avg_reward)
            self.writer.add_scalar('ave_reward/test', avg_reward, self.total_timesteps)
            if self.best_reward < avg_reward:
                self.best_reward = avg_reward
                print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                      (self.total_timesteps, self.episode_timesteps, avg_reward))
                self.policy.save(self.file_name, directory=self.log_dir)
                np.save(self.log_dir + "/test_accuracy", self.evaluations)
                utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
            # else:
            # print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
            #       (total_timesteps, episode_num, episode_timesteps, avg_reward))

    def reset(self):
        # Reset environment
        self.obs = self.env.reset()
        self.episode_reward = 0
        self.episode_progress = 0.0
        self.episode_timesteps = 0

        self.obs_vec = np.dot(np.ones((self.args.seq_len, 1)), self.obs.reshape((1, -1)))

        self.pre_foot_contact = 1
        self.foot_contact = 1
        self.foot_contact_vec = np.asarray([1, 1, 1])
        self.delay_num = self.foot_contact_vec.shape[0] - 1
        self.gait_num = 0
        self.gait_state_mat = np.zeros((0, 9))
        self.idx_angle = np.zeros(0)
        self.reward_angle = np.zeros(0)

        self.still_steps = 0

    def train(self):
        self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
        if 'human_angle' in self.args.method_name:
            human_joint_angle = utils.read_table(file_name=self.project_path + 'data/joint_angle.xls')
        done = True
        while self.total_timesteps < self.args.max_timesteps:
            if done:
                self.train_once()
                self.reset()
                done = False
            # Select action randomly or according to policy
            if self.total_timesteps < self.args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                if 'seq' in self.args.method_name:
                    action = self.policy.select_action(np.array(self.obs_vec))
                else:
                    action = self.policy.select_action(np.array(self.obs))
                if self.args.expl_noise != 0:
                    action = (action + np.random.normal(0, self.args.expl_noise,
                                                        size=self.env.action_space.shape[0])).clip(
                        self.env.action_space.low, self.env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = self.env.step(action)

            self.episode_reward += reward
            self.episode_progress += new_obs[3]

            if 'human_angle' in self.args.method_name:
                reward = self.update_gait_reward(new_obs, reward, human_joint_angle)

            if 'still_steps' in self.args.method_name:
                if np.array_equal(new_obs[-2:], np.asarray([1., 1.])):
                    self.still_steps += 1
                else:
                    self.still_steps = 0
                if self.still_steps > int(400 / self.env_timeStep):
                    self.replay_buffer.add_final_reward(-2.0, self.still_steps - 1)
                    reward -= 2.0
                    done = True

            done_bool = 0 if self.episode_timesteps + 1 == self.env._max_episode_steps else float(done)

            if 'seq' in self.args.method_name:
                # Store data in replay buffer
                new_obs_vec = utils.fifo_data(np.copy(self.obs_vec), new_obs)
                self.replay_buffer.add((np.copy(self.obs_vec), new_obs_vec, action, reward, done_bool))
                self.obs_vec = utils.fifo_data(self.obs_vec, new_obs)
            else:
                self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

            self.obs = new_obs

            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1

        # Final evaluation
        self.evaluations.append(evaluate_policy(self.env, self.policy, self.args))
        np.save(self.log_dir + "/test_accuracy", self.evaluations)
        utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
        self.env.reset()

    def update_gait_reward(self, new_obs, reward, human_joint_angle):
        self.foot_contact_vec = utils.fifo_data(self.foot_contact_vec, new_obs[-2])
        if 0 == np.std(self.foot_contact_vec):
            self.foot_contact = np.mean(self.foot_contact_vec)
        if 1 == (self.foot_contact - self.pre_foot_contact):
            if self.gait_state_mat.shape[0] > int(100 / self.env_timeStep):
                self.gait_num += 1
                if self.gait_num >= 2:
                    # # The ATD3 seems to prefer the negative similarity reward
                    joint_angle_sampled = signal.resample(self.gait_state_mat[:-self.delay_num, 0:6],
                                                          num=human_joint_angle.shape[0])
                    coefficient = utils.calc_cos_similarity(human_joint_angle,
                                                            joint_angle_sampled) - 0.5

                    print('gait_num:', self.gait_num, 'time steps in a gait: ', self.gait_state_mat.shape[0],
                          # 'cross_gait_reward: ', np.round(cross_gait_reward, 2),
                          'coefficient: ', np.round(coefficient, 2),
                          'speed: ', np.round(np.linalg.norm(new_obs[3:6]), 2),
                          'is cross gait: ', utils.check_cross_gait(self.gait_state_mat[:-self.delay_num, :-1]))
                    self.replay_buffer.add_final_reward(coefficient, self.gait_state_mat.shape[0] - self.delay_num,
                                                        delay=self.delay_num)
                    reward_steps = min(int(2000 / self.env_timeStep), len(self.reward_angle))
                    self.replay_buffer.add_specific_reward(self.reward_angle[-reward_steps:],
                                                           self.idx_angle[-reward_steps:])

                self.idx_angle = np.r_[self.idx_angle, self.gait_state_mat[:-self.delay_num, -1]]
                self.reward_angle = np.r_[self.reward_angle,
                                          0.2 * np.ones(self.gait_state_mat[:-self.delay_num, -1].shape[0])]
            self.gait_state_mat = self.gait_state_mat[-self.delay_num:]

        self.pre_foot_contact = self.foot_contact
        gait_state = np.zeros((1, 9))
        gait_state[0, 0:6] = new_obs[8:20:2]
        gait_state[0, 6:-1] = new_obs[-2:]
        gait_state[0, -1] = self.total_timesteps
        self.gait_state_mat = np.r_[self.gait_state_mat, gait_state]
        reward -= 0.5
        return reward

    def eval_only(self):
        video_dir = '{}/video/{}_{}'.format(self.result_path, self.args.env_name, self.args.method_name)
        if self.args.save_video and not os.path.exists(video_dir):
            os.makedirs(video_dir)
        model_path = self.result_path + '/{}/{}_{}'.format(self.args.log_path, self.args.method_name,
                                                           self.args.seed + 1)
        print(model_path)
        self.policy.load("%s" % (self.file_name), directory=model_path)
        for _ in range(10):
            if self.args.save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_name = video_dir + '/{}_{}_{}.mp4'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    self.file_name, self.args.state_noise)
                out_video = cv2.VideoWriter(video_name, fourcc, 60.0, (640, 480))
            obs = self.env.reset()
            if 'seq' in self.args.method_name:
                obs_vec = np.dot(np.ones((self.args.seq_len, 1)), obs.reshape((1, -1)))

            obs_mat = np.asarray(obs)
            done = False

            while not done:
                if 'seq' in self.args.method_name:
                    action = self.policy.select_action(np.array(obs_vec))
                else:
                    action = self.policy.select_action(np.array(obs))

                obs, reward, done, _ = self.env.step(action)

                if 'seq' in self.args.method_name:
                    obs_vec = utils.fifo_data(obs_vec, obs)

                obs[8:20] += np.random.normal(0, self.args.state_noise, size=obs[8:20].shape[0]).clip(
                    -1, 1)
                obs_mat = np.c_[obs_mat, np.asarray(obs)]

                if self.args.save_video:
                    img = self.env.render(mode='rgb_array')
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_video.write(img)
                else:
                    self.env.render()

            if self.args.save_video:
                utils.write_table(video_name + '_state', np.transpose(obs_mat))
                out_video.release()
        self.env.reset()


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
            avg_reward += reward

    avg_reward /= eval_episodes
    # print ("---------------------------------------"                      )
    # print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print ("---------------------------------------"                      )
    return avg_reward