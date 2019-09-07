import argparse
import datetime
import roboschool, gym
import numpy as np
import itertools
import torch
import os
from code.SAC.sac import SAC
from tensorboardX import SummaryWriter
from scipy import signal
from tqdm import tqdm
from code.SAC.replay_memory import ReplayMemory
from code.utils import utils

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="RoboschoolWalker2d-v1",
                    help='name of the environment to run')
parser.add_argument("--method_name", default='human_angle',
                        help='Name of your method (default: )')  # Name of the method
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument("--save_models", default=True)  # Whether or not models are saved
parser.add_argument("--save_video", default=False)
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=3e5, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


file_name = "SAC_%s_%s_%s" % (args.env_name, str(args.seed), args.method_name)
print("---------------------------------------")
print("Settings: %s" % (file_name))
print("---------------------------------------")

result_path = "../../results"
video_dir = '{}/video/SAC_{}_{}'.format(result_path, args.env_name, args.method_name)
model_dir = '{}/models/SAC_master/{}_{}'.format(result_path, args.env_name, args.method_name)
if args.save_models and not os.path.exists(model_dir):
    os.makedirs(model_dir)
if args.save_video and not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#TesnorboardX
writer = SummaryWriter(logdir='../../results/runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

best_reward = 0.0
pbar = tqdm(total=args.num_steps, initial=total_numsteps, position=0, leave=True)
episode_steps = 0
timesteps_since_eval = 0

if 'human_angle' == args.method_name:
    still_steps = 0

    human_joint_angle = utils.read_table()

    pre_foot_contact = 1
    foot_contact = 1
    foot_contact_vec = np.asarray([1, 1, 1])
    delay_num = foot_contact_vec.shape[0] - 1
    gait_num = 0
    joint_angle = np.zeros((0, 7))
    idx_angle = np.zeros(0)
    reward_angle = np.zeros(0)



for i_episode in itertools.count(1):
    pbar.update(episode_steps)

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    if 'human_angle' == args.method_name:
        still_steps = 0
        pre_foot_contact = 1
        foot_contact = 1
        foot_contact_vec = np.asarray([1, 1, 1])
        gait_num = 0
        joint_angle = np.zeros((0, 7))
        idx_angle = np.zeros(0)
        reward_angle = np.zeros(0)

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)
                # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        timesteps_since_eval += 1
        episode_reward += reward

        if 'human_angle' == args.method_name:
            utils.fifo_list(foot_contact_vec, next_state[-2])
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
                        memory.add_final_reward(coefficient, joint_angle.shape[0] - delay_num,
                                                       delay=delay_num)
                        memory.add_specific_reward(reward_angle, idx_angle)
                        idx_angle = np.r_[idx_angle, joint_angle[:-delay_num, -1]]
                        reward_angle = np.r_[reward_angle,
                                             0.05 * np.ones(joint_angle[:-delay_num, -1].shape[0])]
                joint_angle = joint_angle[-delay_num:]
            pre_foot_contact = foot_contact
            joint_angle_obs = np.zeros((1, 7))
            joint_angle_obs[0, :-1] = state[8:20:2]
            joint_angle_obs[-1] = total_numsteps
            joint_angle = np.r_[joint_angle, joint_angle_obs]

            reward -= 0.5

            if np.array_equal(next_state[-2:], np.asarray([1., 1.])):
                still_steps += 1
            else:
                still_steps = 0
            if still_steps > 100:
                memory.add_final_reward(-2.0, still_steps - 1)
                reward -= 2.0
                done = True


        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('ave_reward/train', episode_reward, total_numsteps)
    # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if timesteps_since_eval >= args.eval_freq:
        timesteps_since_eval %= args.eval_freq
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        writer.add_scalar('ave_reward/test', avg_reward, total_numsteps)

        if best_reward < avg_reward:
            best_reward = avg_reward
            print(("Best reward! Total T: %d, Episode Num: %d, Reward: %f") %
                  (total_numsteps, episodes, avg_reward))
            if args.save_models: agent.save(file_name, directory=model_dir)
        else:
            print(("Total T: %d, Episode Num: %d, Reward: %f") %
                  (total_numsteps, episodes, avg_reward))

        # print("----------------------------------------")
        # print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        # print("----------------------------------------")

agent.load("%s" % (file_name), directory=model_dir)
for i in range(1):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(np.array(obs))
        obs, reward, done, _ = env.step(action)
        env.render()


env.close()

