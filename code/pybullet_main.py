import os
print(os.getcwd())
import sys
project_path = '../'
sys.path.insert(0, project_path + 'code')
print(sys.path)
import roboschool, pybullet_envs, gym
import pybullet as p
import argparse
import numpy as np
from utils.solver import utils, Solver

def test_env(env):
    env.reset()
    state = np.random.rand(22)
    print(env.set_robot(state) - state)
    while True:
        env.render()

def main(env, args):
    solver = Solver(args, env, project_path)
    if not args.eval_only:
        solver.train()
        return solver.best_reward
    else:
        solver.eval_only()
        return 1500



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='ATD3_RNN')  # Policy name
    parser.add_argument("--env_name", default="HopperBulletEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/Roboschool_one_step')

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--render", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--video_size", default=(854, 480))
    parser.add_argument("--save_all_policy", default=False)
    parser.add_argument("--load_policy_idx", default='')
    parser.add_argument("--evaluate_Q_value", default=False)
    parser.add_argument("--reward_name", default='r_d_r_s',
                        help='Name of your method (default: )')  # Name of the method

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e5, type=int)  # Max time steps to run environment for

    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    # env_name_vec = ['HopperBulletEnv-v0',
    #                 'HalfCheetahBulletEnv-v0',
    #                 'AntBulletEnv-v0',
    #                 'Walker2DBulletEnv-v0', 'HumanoidBulletEnv-v0', 'InvertedPendulumBulletEnv-v0',
    #                 'InvertedDoublePendulumBulletEnv-v0', 'InvertedPendulumSwingupBulletEnv-v0'
    #                 ]
    env_name_vec = [
    'RoboschoolHopper-v1',
    # 'RoboschoolWalker2d-v1',
    'RoboschoolHalfCheetah-v1',
    'RoboschoolAnt-v1',
    'RoboschoolHumanoid-v1',
    'RoboschoolInvertedPendulum-v1',
    'RoboschoolInvertedPendulumSwingup-v1',
    'RoboschoolInvertedDoublePendulum-v1',
    ]
    policy_name_vec = ['DDPG', 'ATD3', 'ATD3_RNN', 'Average_TD3', 'TD3']
    for env_name in env_name_vec:
        args.env_name = env_name
        env = gym.make(args.env_name)
        if 'Pendulum' in env_name:
            print(env_name)
            args.reward_name = ''
        if args.render and 'Bullet' in env_name:
            env.render('human')
        for policy_name in policy_name_vec:
            args.policy_name = policy_name
            main(env, args)
        env.close()