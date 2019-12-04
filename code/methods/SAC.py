import os
import torch
import torch.nn.functional as F
import glob
from torch.optim import Adam
from utils.utils import soft_update, hard_update
from utils.model import GaussianPolicy, QNetwork, DeterministicPolicy, ValueNetwork


class SAC(object):
    def __init__(self, args, state_dim, action_dim, max_action, action_space):

        self.args = args

        self.alpha = self.args.entropy_alpha
        self.lr = self.args.earning_rate
        self.policy_type = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(state_dim, action_dim, 400).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.value_net = ValueNetwork(state_dim, 400).to(device=self.device)
        self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr)

        self.value_net_target = ValueNetwork(state_dim, 400).to(device=self.device)
        hard_update(self.value_net_target, self.value_net)

        # self.critic_target = QNetwork(state_dim, action_dim, 400).to(self.device)
        # hard_update(self.critic_target, self.critic)
        self.it = 0

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(state_dim, action_dim, 400, max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_dim, action_dim, 400, max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)


    def select_action(self, state, eval=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
        policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        self.it += 1
        # Sample a batch from memory
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(1 - done).to(self.device)

        # Two Q-functions to mitigate positive bias in the policy improvement step
        # Jv(phi) = 0.5(phi(s_t) - (Q(s_t, a_t) - log pi (a_t|s_t)))**2

        pi, log_pi, _ = self.policy.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        phi_val_target = torch.min(qf1_pi, qf2_pi) - self.alpha * log_pi

        phi_val = self.value_net(state)
        phi_loss = F.mse_loss(phi_val, phi_val_target)

        # Jq(theta) = 0.5(q_theta - (r_t + V_target(s_t+1)))**2
        q_val_target = reward + not_done * discount * self.value_net_target(next_state)
        q1, q2 = self.critic(state, action)

        # The default mse_loss reduce to the mean of the element-wise mse loss.
        qf1_loss = F.mse_loss(q1, q_val_target) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(q2, q_val_target) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        # pi, log_pi, _ = self.policy.sample(state)
        # qf1_pi, qf2_pi = self.critic(state, pi)
        # qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (-phi_val_target).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.value_net_optim.zero_grad()
        phi_loss.backward(retain_graph=True)
        self.value_net_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if self.it % self.target_update_interval == 0:
            soft_update(self.value_net_target, self.value_net, tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save(self, filename, directory):
        actor_path = '%s/%s_actor.pth' % (directory, filename)
        critic_path = '%s/%s_critic.pth' % (directory, filename)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        # print('actor path: {}, critic path: {}'.format(actor_path, critic_path))

    def load(self, filename, directory):
        actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
        critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
        print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
        self.policy.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

