import os
print(os.getcwd())
import sys
project_path = '../'
sys.path.insert(0, project_path + 'code')
print(sys.path)
import roboschool, gym
import argparse
from utils.solver import utils, Solver

def main(env, reward_name ='', policy_name ='TD3', state_noise = 0.0, seed = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default=policy_name)  # Policy name
    parser.add_argument("--env_name", default="RoboschoolWalker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/ATD3_walker2d')

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--reward_name", default=reward_name,
                        help='Name of your method (default: )')  # Name of the method

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
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
    reward_name_vec =['r_d', 'r_s', 'r_f', 'r_n', 'r_gv', 'r_lhs', 'r_gs', 'r_cg', 'r_fr', 'r_po']
    # reward_name_vec = ['human_angle_still_steps_seq_ATD3_RNN', 'human_angle_still_steps_ATD3',
    #                    'human_angle_still_steps', 'still_steps', '']
    policy_name_vec = ['ATD3_RNN', 'ATD3', 'TD3']
    env = gym.make('RoboschoolWalker2d-v1')
    for r in range(5, len(reward_name_vec)):
        main(env, reward_name=utils.connect_str_list(reward_name_vec[:r+1]),
             policy_name = policy_name_vec[0])
    for p in range(len(policy_name_vec)):
        main(env, reward_name=utils.connect_str_list(reward_name_vec),
             policy_name = policy_name_vec[p])
    env.close()