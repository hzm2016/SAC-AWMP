import argparse
import datetime
import roboschool
from roboschool import gym_forward_walker, gym_mujoco_walkers
import gym
import numpy as np
import itertools
import torch
from OpenGL import GLU
from tqdm import tqdm
from gym import spaces
from gym import wrappers
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
# from utils import *
from walker2d_utils import *

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="RoboschoolWalker2d-v1",
                    help='name of the environment to run (default: Walker2d-v2, '
                         'Pendulum-v0, RoboschoolWalker2d-v1)')
parser.add_argument('--method_name', default="_final_reward",
                    help='Name of your method (default: )')
parser.add_argument('--result_path', default="../../results",
                    help='path to save model and runs')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--save_video', type=bool, default=False,
                    help='Save video (default:False)')
parser.add_argument('--eval_only', type=bool, default=False,
                    help='Only evaluates a policy without training (default:True)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--model_based', type=bool, default=False,
                    help='Use an impedance control model (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the '
                         'reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=200000, metavar='N',
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

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
if args.save_video:
    env = wrappers.Monitor(env, args.result_path + '/video/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                args.env_name,args.policy,
                                                                "autotune" if args.automatic_entropy_tuning else "")
                           )
print('env setting successfully!')
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

model_based = args.model_based
# Agent
if model_based:
    # action_space = spaces.Box(low=np.array([0]), high=np.array([1e2]), dtype=np.float32)
    low_actions = np.r_[np.zeros(12), -np.ones(6)]
    high_actions = np.r_[1e3 * np.ones(6), 10.0 * np.ones(6), np.ones(6)]
    action_space = spaces.Box(low= low_actions, high=high_actions, dtype=np.float32)
else:
    action_space = env.action_space
# print(action_space)
agent = SAC(env.observation_space.shape[0], action_space, args)
if not args.eval_only:
    #TesnorboardX
    writer = SummaryWriter(logdir= args.result_path + '/runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                args.env_name,args.policy,
                                                                "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    pre_num_steps = 0
    updates = 0

    best_reward = -1e5
    longest_x_dist = 0.0
    pbar = tqdm(total=args.num_steps, initial=total_numsteps)
    for i_episode in itertools.count(1):
        pbar.update(total_numsteps - pre_num_steps)
        pre_num_steps = total_numsteps
        episode_reward = 0
        episode_my_reward = 0.0
        episode_steps = 0
        done = False
        state = env.reset()

        # # state[0:6] = [x, y, z,
        # # 0.3*vx, 0.3*vy, 0.3*vz]
        # # state[6:10] = [np.sin(self.angle_to_target), np.cos(self.angle_to_target),
        # # row, pitch]
        # # state[10:22:2]: joint position, scaled to -1..+1 between limits
        # # state[11:22:2]: joint speed, scaled to -1..+1 between limits
        # # state[22:24]: right / left foot state (touch = 1.0)

        # state[0:8] = np.array([
        #             x, y, z,
        #             0.3*vx, 0.3*vy, 0.3*vz,    # 0.3 is just scaling typical speed into -1..+1, no physical sense here
        #             r, p], dtype=np.float32)
        # state[8:20:2]: joint position, scaled to -1..+1 between limits
        # state[9:20:2]: joint speed, scaled to -1..+1 between limits
        # state[20:22]: right / left foot state (touch = 1.0)



        # pre_state = np.copy(state)
        # last_phase_state = None
        # last_phase_action = None

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            # xyz = state[0:3]
            # v_xyz = state[3:6]
            # my_reward = 1.0
            # if v_xyz[0] > 0.2:
            #     my_reward += 2.0
            # else:
            #     my_reward -= -2.0
            #
            # pitch = state[7]
            # if abs(xyz[2]) > 0.8 or abs(pitch) > 1.0:
            #     my_reward = -2.0
            #     done = True

            # env.render()
            # print('xyz ', state[0:3])
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            state = next_state
        # while not done:
        #     if len(memory) > args.batch_size:
        #         # Number of updates per step in environment
        #         for i in range(args.updates_per_step):
        #             # Update parameters of all the networks
        #             critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
        #                 agent.update_parameters(memory, args.batch_size, updates)
        #
        #             writer.add_scalar('loss/critic_1', critic_1_loss, updates)
        #             writer.add_scalar('loss/critic_2', critic_2_loss, updates)
        #             writer.add_scalar('loss/policy', policy_loss, updates)
        #             writer.add_scalar('loss/entropy_loss', ent_loss, updates)
        #             writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        #             updates += 1
        #
        #     if model_based:
        #         if args.start_steps > total_numsteps:
        #             action = random_action(action_space)
        #         else:
        #             action = agent.select_action(state)  # Sample action from policy
        #         torque, theta = calc_torque(state, action)
        #         last_phase_state = np.copy(state)
        #         last_phase_action = np.copy(action)
        #
        #     if not model_based:
        #         if args.start_steps > total_numsteps:
        #             action = env.action_space.sample()  # Sample random action
        #         else:
        #             action = agent.select_action(state)  # Sample action from policy
        #         torque = action
        #
        #     v_xyz = state[3:6]
        #
        #     # if v_xyz[0] > 0.5:
        #     #     velocity_reward = 1.0
        #     # else:
        #     #     velocity_reward = -1.0
        #
        #     # # calculate positive energy
        #     # positive_energy_reward = torque * joing_speed
        #     # positive_energy_reward = -2e-4 * np.sum(positive_energy_reward.clip(min=0))
        #     #
        #     # pre_xyz = pre_state[0:3]
        #     # move_reward = np.clip(xyz[0] - pre_xyz[0], a_min = -0.5, a_max=0.5)
        #
        #
        #     next_state, reward, done, _ = env.step(torque) # Step
        #     episode_steps += 1
        #     total_numsteps += 1
        #
        #     # Ignore the "done" signal if it comes from hitting the time horizon.
        #     # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        #     mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        #
        #     # my_reward = reward
        #
        #     memory.push(state, action, reward, next_state, mask) # Append transition to memory
        #
        #     pre_state[:] = state[:]
        #     state[:] = next_state[:]
        #
        #
        #     episode_reward += reward
        #     # episode_my_reward += my_reward

        if total_numsteps > args.num_steps:
            break
        # if state[0] > 1:
        #     final_reward = 2 * state[0]
        # else:
        #     final_reward = 0.0
        # memory.add_final_reward(final_reward=final_reward, steps=episode_steps)

        writer.add_scalar('reward/train', episode_reward, i_episode)
        # writer.add_scalar('my_reward/train', episode_my_reward, i_episode)
        # print("Episode: {}, episode_reward: {}, my_reward: {}, positive_energy_reward: {}".format(
        #     i_episode, episode_reward, my_reward, positive_energy_reward))

        if args.eval == True:
            episodes = 10
            avg_reward, ave_x_dist = eval_agent(env, agent, model_based=model_based)
            # print("Test Episodes: {}, Reward: {}".format(i_episode, avg_reward))
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            writer.add_scalar('ave_x_dist/test', ave_x_dist, i_episode)

            if (ave_x_dist > longest_x_dist) and (total_numsteps > args.start_steps):
                agent.save_model(args.result_path, args.env_name + args.method_name)
                # best_reward = avg_reward
                longest_x_dist = ave_x_dist
                if ave_x_dist > 1.0:
                    render_env(env, agent, model_based=model_based)
                print("----------------------------------------")
                print("Test Episodes: {}, Longest_x_dist: {}".format(
                    episodes, round(longest_x_dist, 2)))
                print("----------------------------------------")
else:
    agent.load_model(args.result_path, args.env_name + args.method_name)
    # episodes = 100
    # avg_reward = eval_agent(env, agent, model_based=model_based, episodes=episodes)
    # print("Final test episodes: {}, Reward: {}".format(episodes, round(avg_reward, 2)))
    for i in range(5):
        render_env(env, agent, model_based = model_based, save_video =args.save_video)
        # render_env(env, agent, k = 100.0, b = 0.5, k_g=20.0, model_based = model_based)
env.close()

