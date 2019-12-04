import numpy as np
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import torch.nn.functional as F
from utils.utils import soft_update, hard_update
from torch.distributions import Categorical
from utils.model import GaussianPolicy1D, ActorList, Critic, Option, ValueNetwork
if torch.cuda.is_available():
	torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HRLSAC(object):
	def __init__(self, args, state_dim, action_dim, max_action,
				 entropy_coeff=0.1, c_reg=1.0, c_ent=4,
				 action_noise=0.2, policy_noise=0.2, noise_clip=0.5,
				 alpha=0.05, option_buffer_size=5000,
				 option_num=2, weighted_action=True):

		self.args = args

		self.actor = GaussianPolicy1D(state_dim, action_dim, max_action, self.args.option_num).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.learning_rate)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.critic_target = Critic(state_dim, action_dim).to(device)
		hard_update(self.critic_target, self.critic)

		self.value_net = ValueNetwork(state_dim, 400).to(device=device)
		self.value_net_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.args.learning_rate)

		self.value_net_target = ValueNetwork(state_dim, 400).to(device=device)
		hard_update(self.value_net_target, self.value_net)

		self.option = Option(state_dim, action_dim, self.args.option_num).to(device)
		self.option_optimizer = torch.optim.Adam(self.option.parameters(), lr=self.args.learning_rate)

		self.max_action = max_action
		self.it = 0

		self.entropy_coeff = entropy_coeff
		self.c_reg = c_reg
		self.c_ent = c_ent

		self.option_buffer_size = self.args.option_buffer_size
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.option_num = self.args.option_num
		self.action_noise = action_noise
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.q_predict = np.zeros(self.option_num)
		self.option_val = 0
		self.alpha = self.args.entropy_alpha
		self.weighted_action = self.args.weighted_action

	def train(self, replay_buffer, discount=0.99, tau=0.005, policy_freq=1):

		self.it += 1
		state, action, target_q, phi_val_target, _, sampling_prob = \
			self.calc_target_phi(replay_buffer, self.args.critic_batch_size, discount, is_on_poliy=False)

		# ================ Train the critic =============================================#
		self.train_critic(state, action, target_q)
		# ===============================================================================#

		# ================ Train the value net =============================================#
		self.train_value_net(state, phi_val_target)

		if self.it % policy_freq == 0:
			# ================ Train the actor =============================================#
			self.train_actor(phi_val_target)
			# ===============================================================================#
			# Update the frozen target models
			for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
		# ==================================================================================#

		# Delayed policy updates
		# if self.it % policy_freq == 0:
			# # Compute actor loss
			# x, y, u, r, d, p = replay_buffer.sample(batch_size)
			# state = torch.FloatTensor(x).to(device)
			# action = torch.FloatTensor(u).to(device)
			# _, _, option_estimated, _ = self.option(state, action)
			# p_normalized = option_estimated / torch.sum(option_estimated, dim=1, keepdim=True)
			# action_list, log_pi_list, _ = self.actor.sample(state)
			# # (batch_num, action_dim, option_num) x (batch_num, option, 1) -> (batch_num, action_dim)
			# if self.weighted_action:
			# 	action = torch.matmul(action_list, p_normalized[:, :, None])[:, :, 0]
			# 	log_pi = torch.matmul(log_pi_list, p_normalized[:, :, None])[:, :, 0]
			# else:
			# 	max_option_idx = torch.argmax(option_estimated, dim=1)
			# 	action = action_list[torch.arange(state.shape[0]), :, max_option_idx]
			# 	log_pi = log_pi_list[torch.arange(state.shape[0]), :, max_option_idx]

		# Delayed option updates
		if self.it % self.option_buffer_size == 0:
			for _ in range(int(self.option_buffer_size)):
				state, action, target_q, predicted_v, sampling_prob = \
					self.calc_target_q(replay_buffer, self.args.option_batch_size, discount, is_on_poliy=True)
				for _ in range(int(self.option_num)):
					self.train_option(state, action, target_q, predicted_v, sampling_prob)
			# ===============================================================================#

	def train_critic(self, state, action, target_q):
		'''
		Calculate the loss of the critic and train the critic.
		'''
		current_q1, current_q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_q1, target_q) + \
					  F.mse_loss(current_q2, target_q)

		# Three steps of training net using PyTorch:
		self.critic_optimizer.zero_grad()  # 1. Clear cumulative gradient
		critic_loss.backward()  # 2. Back propagation
		self.critic_optimizer.step()  # 3. Update the parameters of the net

	def train_value_net(self, state, phi_val_target):
		phi_val = self.value_net(state)
		phi_loss = F.mse_loss(phi_val, phi_val_target)

		# Three steps of training net using PyTorch:
		self.value_net_optim.zero_grad()  # 1. Clear cumulative gradient
		phi_loss.backward(retain_graph=True)  # 2. Back propagation
		self.value_net_optim.step()  # 3. Update the parameters of the net

	def train_actor(self, phi_val_target):
		'''
		Calculate the loss of the actor and train the actor
		'''
		# current_q1, current_q2 = self.critic(state, action)
		# actor_loss = ((self.alpha * log_pi) - current_q1).mean()
		actor_loss = (- phi_val_target).mean()

		# Optimize the actor
		# Three steps of training net using PyTorch:
		self.actor_optimizer.zero_grad()  # 1. Clear cumulative gradient
		actor_loss.backward()  # 2. Back propagation
		self.actor_optimizer.step()  # 3. Update the parameters of the net

	def train_option(self, state, action, target_q, predicted_v, sampling_prob):
		xu, decoded_xu, output_option, output_option_noise = self.option(state, action)
		advantage = target_q - predicted_v

		weight = torch.exp(advantage - torch.max(advantage)) / sampling_prob
		w_norm = weight / torch.mean(weight)

		critic_conditional_entropy = weighted_entropy(output_option, w_norm)
		p_weighted_ave = weighted_mean(output_option, w_norm)
		critic_entropy = critic_conditional_entropy - self.c_ent * entropy(p_weighted_ave)

		vat_loss = kl(output_option, output_option_noise)

		reg_loss = F.l1_loss(xu, decoded_xu)
		option_loss = reg_loss + self.entropy_coeff * critic_entropy + self.c_reg * vat_loss

		# Optimize the option
		# Three steps of training net using PyTorch:
		self.option_optimizer.zero_grad()  # 1. Clear cumulative gradient
		option_loss.backward(retain_graph=True)  # 2. Back propagation
		self.option_optimizer.step()  # 3. Update the parameters of the net

	def calc_target_phi(self, replay_buffer, batch_size=100, discount=0.99, is_on_poliy=True):
		policy_noise = self.policy_noise
		noise_clip = self.noise_clip

		if is_on_poliy:
			x, y, u, r, d, p = \
				replay_buffer.sample_on_policy(batch_size, self.option_buffer_size)
		else:
			x, y, u, r, d, p = \
				replay_buffer.sample(batch_size)

		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		next_state = torch.FloatTensor(y).to(device)
		not_done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)
		sampling_prob = torch.FloatTensor(p).to(device)

		action_list, log_pi_list, _ = self.actor.sample(state)
		_, _, option_estimated, _ = self.option(state, action)
		p_normalized = option_estimated / torch.sum(option_estimated, dim=1, keepdim=True)

		# (batch_num, action_dim, option_num) x (batch_num, option, 1) -> (batch_num, action_dim)
		if self.weighted_action:
			action_pi = torch.matmul(action_list, p_normalized[:, :, None])[:, :, 0]
			log_pi = torch.matmul(log_pi_list, p_normalized[:, :, None])[:, :, 0]
		else:
			max_option_idx = torch.argmax(p_normalized, dim=1)
			action_pi = action_list[torch.arange(state.shape[0]), :, max_option_idx]
			log_pi = log_pi_list[torch.arange(state.shape[0]), :, max_option_idx]

		# option_batch, _, q_predict, p_normalized = self.softmax_option_target(state)
		# # (batch_num, action_dim, option_num) x (batch_num, option, 1) -> (batch_num, action_dim)
		# if self.weighted_action:
		# 	action_pi = torch.matmul(action_list, p_normalized[:, :, None])[:, :, 0]
		# 	log_pi = torch.matmul(log_pi_list, p_normalized[:, :, None])[:, :, 0]
		# else:
		# 	action_pi = action_list[torch.arange(state.shape[0]),:,option_batch]
		# 	log_pi = log_pi_list[torch.arange(state.shape[0]),:,option_batch]

		qf1_pi, qf2_pi = self.critic(state, action_pi)
		phi_val_target = torch.min(qf1_pi, qf2_pi) - self.alpha * log_pi

		target_q = reward + not_done * discount * self.value_net_target(next_state)

		predicted_v = self.value_func(state)

		# predicted_v = self.value_net_target(state)
		return state, action, target_q, phi_val_target, predicted_v, sampling_prob

	def calc_target_q(self, replay_buffer, batch_size=100, discount=0.99, is_on_poliy=True):
		if is_on_poliy:
			x, y, u, r, d, p = \
				replay_buffer.sample_on_policy(batch_size, self.option_buffer_size)
		else:
			x, y, u, r, d, p = \
				replay_buffer.sample(batch_size)

		state = torch.FloatTensor(x).to(device)
		next_state = torch.FloatTensor(y).to(device)
		action = torch.FloatTensor(u).to(device)
		not_done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)
		sampling_prob = torch.FloatTensor(p).to(device)

		# target_q = reward + not_done * discount * self.value_net_target(next_state)
		q_predict_1, q_predict_2 = self.critic_target(state, action)

		target_q = torch.min(q_predict_1, q_predict_2)

		predicted_v = self.value_func(state)

		# predicted_v = self.value_net_target(state)
		return state, action, target_q, predicted_v, sampling_prob

	def value_func(self, states):
		batch_size = states.shape[0]
		action_list, _, _ = self.actor.sample(states)
		option_num = action_list.shape[-1]

		# action: (batch_num, action_dim, option_num)-> (batch_num, option_num, action_dim)
		# -> (batch_num * option_num, action_dim)
		action_list = action_list.transpose(1, 2)
		action_list = action_list.reshape((-1, action_list.shape[-1]))

		# states: (batch_num, state_dim) -> (batch_num, state_dim * option_num)
		# -> (batch_num * option_num, state_dim)
		states = states.repeat(1, option_num).view(batch_size * option_num, -1)
		q_predict_1, q_predict_2 = self.critic_target(states, action_list)

		# q_predict: (batch_num * option_num, 1) -> (batch_num, option_num)
		q_predict = torch.min(q_predict_1, q_predict_2).view(batch_size, -1)
		po = softmax(q_predict)

		return weighted_mean_array(q_predict, po)

	def softmax_option_target(self, states):

		# Q_predict_i: B*O， B: batch number, O: option number
		batch_size = states.shape[0]
		action_list, _, _ = self.actor.sample(states)  # (batch_num, action_dim, option_num)
		option_num = action_list.shape[-1]

		# action: (batch_num, action_dim, option_num)-> (batch_num, option_num, action_dim)
		# -> (batch_num * option_num, action_dim)
		action_list = action_list.transpose(1, 2)
		action_list = action_list.reshape((-1, action_list.shape[-1]))

		# states: (batch_num, state_dim) -> (batch_num, state_dim * option_num)
		# -> (batch_num * option_num, state_dim)
		states = states.repeat(1, option_num).view(batch_size * option_num, -1)

		q_predict_1, q_predict_2 = self.critic_target(states, action_list)

		# q_predict: (batch_num * option_num, 1) -> (batch_num, option_num)
		# q_predict = (0.5 * (q_predict_1 + q_predict_2)).view(batch_size, -1)
		q_predict = q_predict_1.view(batch_size, -1)

		p = softmax(q_predict)
		p_sum = torch.sum(p, dim=1, keepdim=True)
		p_normalized = p / p_sum
		o_softmax = p_sample(p)

		if self.weighted_action:
			q_softmax = torch.matmul(q_predict[:, None, :], p_normalized[:, :, None])[:, 0, 0]
		else:
			q_softmax = q_predict[:, o_softmax]

		return o_softmax, q_softmax, q_predict, p_normalized

	def select_action(self, state, eval=True):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		# select option
		option_batch, _, q_predict, p_normalized = self.softmax_option_target(state)

		if eval == False:
			action_list, _, _ = self.actor.sample(state)
		else:
			option_batch = torch.argmax(q_predict, dim=-1)
			_, _, action_list = self.actor.sample(state)

		# The weighted sum of the action
		# (batch_num, action_dim, option_num) x (batch_num, option, 1) -> (batch_num, action_dim)
		if self.weighted_action:
			action = torch.matmul(action_list, p_normalized[:, :, None])[:, :, 0]
			# print('weighted', p_normalized[:, :, None])
			# print('action_list', action_list)
			# print('action', action)
		else:
			# Below is to select action based on the option
			action = action_list[torch.arange(state.shape[0]), :, option_batch]

		self.q_predict = q_predict.cpu().data.numpy().flatten()
		self.option_val = option_batch.cpu().data.numpy().flatten()
		return action.cpu().data.numpy().flatten()

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))

	@staticmethod
	def calc_actor_discrepancy(state, actor):
		action_tensor = actor(state)
		action_discrepancy = action_tensor - torch.mean(action_tensor, dim=-1, keepdim=True)
		action_mse = torch.mean(torch.sum(action_discrepancy ** 2, dim=-1))
		return action_mse / (torch.abs(action_mse) + 1)

	def cal_estimate_value(self, replay_buffer, eval_states=10000):
		x, _, u, _, _ = replay_buffer.sample(eval_states)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		Q1, Q2 = self.critic(state, action)
		# target_Q = torch.mean(torch.min(Q1, Q2))
		Q_val = 0.5 * (torch.mean(Q1) + torch.mean(Q2))
		return Q_val.detach().cpu().numpy()


def add_randn(x_input, vat_noise):
	"""
	add normal noise to the input
    """
	epsilon = torch.FloatTensor(torch.randn(size=x_input.size())).to(device)
	return x_input + vat_noise * epsilon * torch.abs(x_input)


def entropy(p):
	return torch.sum(p * torch.log((p + 1e-8)))


def kl(p, q):
	return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)))


def p_sample(p):
	'''
    :param p: size: (batch_size, option)
    :return: o_softmax: (batch_size)
    '''
	p_sum = torch.sum(p, dim=1, keepdim=True)
	p_normalized = p / p_sum
	m = Categorical(p_normalized)
	return m.sample()


def softmax(x):
	# This function is different from the Eq. 17, but it does not matter because
	# both the nominator and denominator are divided by the same value.
	# Equation 17: pi(o|s) = ext(Q^pi - max(Q^pi))/sum(ext(Q^pi - max(Q^pi))
	x_max, _ = torch.max(x, dim=1, keepdim=True)
	e_x = torch.exp(x - x_max)
	e_x_sum = torch.sum(e_x, dim=1, keepdim=True)
	out = e_x / e_x_sum
	return out


def weighted_entropy(p, w_norm):
	return torch.sum(w_norm * p * torch.log(p + 1e-8))


def weighted_mean(p, w_norm):
	return torch.mean(w_norm * p, axis=0)


def weighted_mean_array(x, weights):
	weights_mean = torch.mean(weights, dim=1, keepdim=True)
	x_weighted = x * weights
	mean_weighted = torch.mean(x_weighted, dim=1, keepdim=True) / weights_mean
	return mean_weighted
