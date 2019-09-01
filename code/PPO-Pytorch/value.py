import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
	def __init__(self, s_dim):
		super(Value, self).__init__()

		self.net = nn.Sequential(nn.Linear(s_dim, 64),
		                         nn.ReLU(),
		                         nn.Linear(64, 64),
		                         nn.ReLU(),
		                         nn.Linear(64, 1))

	def forward(self, s):
		"""

		:param s: [b, s_dim]
		:return:  [b, 1]
		"""
		value = self.net(s)

		return value
