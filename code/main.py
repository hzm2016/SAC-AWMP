import os
print(os.getcwd())
import sys
project_path = '../'
sys.path.insert(0, project_path + 'code')
print(sys.path)
import roboschool, gym
from roboschool import gym_forward_walker, gym_atlas
import argparse
import numpy as np
from utils.solver import utils, Solver

def test_env(env):
    env.reset()
    state = np.random.rand(22)
    # state = np.asarray([0.24301213, 0.93942997, 0.03958391, 0.93737306, 0.9552348, 0.72928275
    #                        , 0.70990771, 0.28037012, 0.72994741, 0.5695038, 0.62305463, 0.19212616
    #                        , 0.01702465, 0.00437125, 0.85252375, 0.97369661, 0.94323611, 0.76765484
    #                        , 0.44252152, 0.32101413, 0.09566902, 0.84304301])
    # state = np.asarray([0.24301213, 0.93942997, 0.03958391,
    #                     0.0,  0.0,   0.0,
    #                     0.70990771, 0.0,
    #                     -0.78 , 0.8,  -0.15,   0.6, -0.49, 0.89,
    #                     -0.51, 0.78, -0.05, 0.78, -0.23, 0.56,
    #                     0.1, 0.3])
    # print(state)
    print(env.set_robot(state) - state)
    while True:
        env.render()

def main(env, reward_name ='', policy_name ='TD3', state_noise = 0.0, seed = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default=policy_name)  # Policy name
    parser.add_argument("--env_name", default="RoboschoolWalker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/ATD3_walker2d_Q_value')

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--render", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--evaluate_Q_value", default=True)
    parser.add_argument("--reward_name", default=reward_name,
                        help='Name of your method (default: )')  # Name of the method

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e5, type=float)  # Max time steps to run environment for

    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=state_noise, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    solver = Solver(args, env, project_path)
    if not args.eval_only:

        solver.train()
    else:
        solver.eval_only()


if __name__ == "__main__":
    '''
    Reward ablation study：default reward (r_d), still_steps (r_s), final speed (r_f), 
    gait num (r_n), gait velocity (r_gv), left heel strike (r_lhs), 
    gait symmetry (r_gs),  cross gait (r_cg), foot recovery (r_fr), push off (r_po)
    '''
    reward_name_vec =['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    policy_name_vec = ['TD3', 'ATD3', 'ATD3_RNN']
    env = gym.make('RoboschoolWalker2d-v1')
    # env = gym.make('RoboschoolHalfCheetah-v1')
    # for r in [0, 4]:
    #     main(env, reward_name=utils.connect_str_list(reward_name_vec[:r+1]),
    #          policy_name = policy_name_vec[0])
    for p in [0, 1, 2]:
        main(env, reward_name=utils.connect_str_list([reward_name_vec[0]]),
             policy_name = policy_name_vec[p])
    # main(env, reward_name=utils.connect_str_list([reward_name_vec[0]]),
    #      policy_name=policy_name_vec[0])
    # test_env(env)
    env.close()
