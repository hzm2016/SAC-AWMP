import time, os
import roboschool
import argparse
import torch
import gym
from OpenGL import GLU
from ppo import PPO
from tqdm import trange

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--eval_only', type=bool, default=False,
                    help='Only evaluates a policy without training (default:True)')


def make_env():
	# env = gym.make('Humanoid-v1')
	env = gym.make('RoboschoolWalker2d-v1')
	return env


def main(args):
	torch.set_default_tensor_type('torch.DoubleTensor')

	batchsz = 2048
	ppo = PPO(make_env, 10)
	if args.eval_only:
		# load model from checkpoint
		ppo.load()
		# comment this line to close evaluaton thread, to speed up training process.
		ppo.render(2)
	else:
		for i in trange(10000):
			ppo.update(batchsz)
			if i % 100 == 0 and i:
				ppo.save()


if __name__ == '__main__':
	print('make sure to execute: [export OMP_NUM_THREADS=1] already.')
	args = parser.parse_args()
	main(args)
