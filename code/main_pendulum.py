import argparse
import datetime
# import roboschool
import gym
import numpy as np
import itertools
import torch
from tqdm import tqdm
from gym import spaces
from gym import wrappers
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from utils import *


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="Pendulum-v0",
                    help='name of the environment to run (default: Walker2d-v2, '
                         'Pendulum-v0, RoboschoolWalker2d-v1)')
parser.add_argument('--method_name', default="_low_energy",
                    help='Name of your method (default: )')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--save_video', type=bool, default=False,
                    help='Save video (default:False)')
parser.add_argument('--eval_only', type=bool, default=True,
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
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
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

model_based = args.model_based
# Agent
if model_based:
    # action_space = spaces.Box(low=np.array([0]), high=np.array([1e2]), dtype=np.float32)
    action_space = spaces.Box(low= np.array([-1e2, -1e1]), high=np.array([1e2, 1e1]), dtype=np.float32)
else:
    action_space = env.action_space
# print(action_space)
agent = SAC(env.observation_space.shape[0], action_space, args)

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
        state = env.reset()
        pre_num_steps = total_numsteps
        while not done:
            if len(memory) > args.batch_size:
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
                if model_based:
                    action = random_action(action_space)
                else:
                    action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy
            if model_based:
                # torque, theta = calc_torque(state, k=abs(action[0]), b=0.5, k_g=20.0)
                torque, theta = calc_torque(state, k = action[0], b = action[1], k_g = 20.0)
            else:
                torque = action

            # calculate positive energy
            positive_energy = torque[0] * state[-1]
            if positive_energy < 0:
                positive_energy == 0

            next_state, reward, done, _ = env.step(torque) # Step
            episode_steps += 1
            total_numsteps += 1

            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            reward_all = reward - args.alpha * positive_energy
            memory.push(state, action, reward_all, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        writer.add_scalar('positive_energy/train', positive_energy, i_episode)
        # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
        #     i_episode, total_numsteps, episode_steps, episode_reward))

        if args.eval == True:
            episodes = 10
            avg_reward = eval_agent(env, agent, model_based=model_based)
            # print("Test Episodes: {}, Reward: {}".format(i_episode, avg_reward))
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            if avg_reward > best_reward:
                agent.save_model(args.env_name + args.method_name)
                best_reward = avg_reward
                render_env(env, agent, model_based=model_based)
                print("----------------------------------------")
                print("Test Episodes: {}, Best reward: {}, Action: {}".format(
                    episodes, round(best_reward, 2), action))
                print("----------------------------------------")
else:
    agent.load_model(args.env_name + args.method_name)
    episodes = 100
    avg_reward = eval_agent(env, agent, model_based=model_based, episodes=episodes)
    print("Final test episodes: {}, Reward: {}".format(episodes, round(avg_reward, 2)))
    for i in range(5):
        render_env(env, agent, model_based = model_based, save_video =args.save_video)
        # render_env(env, agent, k = 100.0, b = 0.5, k_g=20.0, model_based = model_based)
env.close()

