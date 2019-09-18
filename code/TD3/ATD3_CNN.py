import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, seq_len = 1):
		super().__init__()
		# out_len = (in_len + 2 * padding - kernel_size) / stride + 1
		self.conv1 = nn.Conv2d(1, 400, kernel_size=[seq_len, state_dim], stride=[1, 1], padding=[0, 0]) # out: (1, 1)

		self.fc1 = nn.Linear(400, 300)
		self.fc2 = nn.Linear(300, action_dim)
		self.max_action = max_action

	def forward(self, x):

		x = x.reshape((-1, 1, x.shape[-2], x.shape[-1]))
		x = F.relu(self.conv1(x))
		x = x.view(-1, 400)
		x = F.relu(self.fc1(x))
		x = self.max_action * torch.tanh(self.fc2(x))
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, seq_len = 1):
		super(Critic, self).__init__()
		# Q1 architecture
		self.conv1 = nn.Conv2d(1, 300, kernel_size=[seq_len, state_dim], stride=[1, 1], padding=[0, 0])  # out: (1, 1)
		# self.bn1 = nn.BatchNorm2d(300)
		self.l1 = nn.Linear(action_dim, 100)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.conv2 = nn.Conv2d(1, 300, kernel_size=[seq_len, state_dim], stride=[1, 1], padding=[0, 0])  # out: (1, 1)
		# self.bn2 = nn.BatchNorm2d(300)
		self.l4 = nn.Linear(action_dim, 100)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = x.reshape((-1, 1, x.shape[-2], x.shape[-1]))
		xh1 = F.relu(self.conv1(x))
		xh1 = xh1.view(-1, 300)
		u1 = F.relu(self.l1(u))
		x1 = torch.cat([xh1, u1], 1)

		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		xh2 = F.relu(self.conv2(x))
		xh2 = xh2.view(-1, 300)
		u2 = F.relu(self.l4(u))
		x2 = torch.cat([xh2, u2], 1)

		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2


	def Q1(self, x, u):
		x = x.reshape((-1, 1, x.shape[-2], x.shape[-1]))
		xh1 = F.relu(self.bn1(self.conv1(x)))
		xh1 = xh1.view(-1, 300)
		u1 = F.relu(self.l1(u))
		x1 = torch.cat([xh1, u1], 1)

		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1 


class ATD3_CNN(object):
	def __init__(self, state_dim, action_dim, max_action, seq_len):
		self.actor = Actor(state_dim, action_dim, max_action, seq_len).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action, seq_len).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim, seq_len).to(device)
		self.critic_target = Critic(state_dim, action_dim, seq_len).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.max_action = max_action


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(-1, state.shape[0], state.shape[1])).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Select action according to policy and add clipped noise 
			noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			# if torch.rand(1) > 0.5:
			# 	target_Q = target_Q1
			# else:
			# 	target_Q = target_Q2

			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) - \
						  0.1 * F.mse_loss(current_Q1, current_Q2)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:

				# Compute actor loss
				current_Q1, current_Q2 = self.critic(state, self.actor(state))
				actor_loss = -0.5 * (current_Q1 + current_Q2).mean()

				# actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
				
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
