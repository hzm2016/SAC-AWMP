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
parser.add_argument('--method_name', default="_finite_phases",
                    help='Name of your method (default: )')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--save_video', type=bool, default=False,
                    help='Save video (default:False)')
parser.add_argument('--eval_only', type=bool, default=False,
                    help='Only evaluates a policy without training (default:True)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
# parser.add_argument('--model_based', type=bool, default=True,
#                     help='Use an impedance control model (default:True)')
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
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=200000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
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
    env = wrappers.Monitor(env,'../results/video/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                args.env_name,args.policy,
                                                                "autotune" if args.automatic_entropy_tuning else "")
                           # ,video_callable=lambda x: True
                           # ,resume=True
                           )
print('env setting successfully!')
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
low_actions = np.r_[np.zeros(12), -np.ones(6)]
high_actions = np.r_[5e2 * np.ones(6), 10.0 * np.ones(6), np.ones(6)]
action_space = spaces.Box(low= low_actions, high=high_actions, dtype=np.float32)
# print(action_space)
agent = SAC(env.observation_space.shape[0] + 2, action_space, args)
time_step = 0.0165/4.0
if not args.eval_only:
    #TesnorboardX
    writer = SummaryWriter(logdir='../results/runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                args.env_name,args.policy,
                                                                "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    pre_num_steps = 0
    updates = 0

    best_reward = -1e5
    pbar = tqdm(total=args.num_steps, initial=total_numsteps)
    for i_episode in itertools.count(1):
        pbar.update(total_numsteps - pre_num_steps)
        episode_reward = 0
        episode_steps = 0
        done = False
        # np.array([np.cos(theta), np.sin(theta), thetadot])
        # state[0:6] = [x, y, z,
        # 0.3*vx, 0.3*vy, 0.3*vz]
        # state[6:10] = [np.sin(self.angle_to_target), np.cos(self.angle_to_target),
        # row, pitch]
        # state[10:22:2]: joint position, scaled to -1..+1 between limits
        # state[11:22:2]: joint speed, scaled to -1..+1 between limits
        # state[22:24]: right / left foot state (touch = 1.0)
        state = env.reset()
        action = agent.select_action(state)
        pre_num_steps = total_numsteps
        pre_state = np.copy(state)
        last_phase_state = np.copy(state)
        last_phase_action = np.copy(action)
        phase_reward = 0.0
        phase_time = 0.0
        while not done:
            is_same_phase = np.array_equal(pre_state[-2:], state[-2:])
            if not is_same_phase:
                # if phase_time < 0.1: # switch foot too frequently
                #     phase_reward += 25 * (state[0] - last_phase_state[0]) - 250 * (0.1 - phase_time)
                # else: # average speed
                phase_reward += 25.0 * (state[0] - last_phase_state[0])

                phase_reward *= 1e-1
                memory.push(last_phase_state, last_phase_action, phase_reward,
                            state, mask)  # Append transition to memory
                if len(memory) > args.batch_size and not is_same_phase:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                            agent.update_parameters(memory, args.batch_size, updates)

                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1

                if args.start_steps > total_numsteps:
                    action = random_action(action_space)
                else:
                    action = agent.select_action(state)  # Sample action from policy

                # change phase variables
                phase_reward = 0.0
                phase_time = 0.0
                last_phase_state[:] = state[:]
                last_phase_action[:] = action[:]

            torque, theta = calc_torque(state, action)
            next_state, reward, done, _ = env.step(torque) # Step

            episode_steps += 1
            total_numsteps += 1

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            # my_reward = alive_reward + move_reward
            phase_reward += reward
            phase_time += time_step

            pre_state[:] = state[:]
            state[:] = next_state[:]
            episode_reward += reward

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        writer.add_scalar('reward/train_each_phase', phase_reward, i_episode)

        if (args.eval == True):
            episodes = 10
            avg_reward = eval_agent_phase(env, agent)
            # print("Test Episodes: {}, Reward: {}".format(i_episode, avg_reward))
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            if (avg_reward > best_reward) and (args.start_steps < total_numsteps):
                agent.save_model(args.env_name + args.method_name)
                best_reward = avg_reward
                if best_reward > 0:
                    render_env_phase(env, agent)
                print("----------------------------------------")
                print("Test Episodes: {}, Best reward: {}".format(
                    episodes, round(best_reward, 2)))
                print("----------------------------------------")
else:
    agent.load_model(args.env_name + args.method_name)
    # episodes = 100
    # avg_reward = eval_agent(env, agent, model_based=model_based, episodes=episodes)
    # print("Final test episodes: {}, Reward: {}".format(episodes, round(avg_reward, 2)))
    for i in range(5):
        render_env_phase(env, agent, save_video =args.save_video)
env.close()

