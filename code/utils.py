import math
import torch
import numpy as np
from tqdm import trange


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def random_action(action_space):
    action = []
    for i in range(action_space.low.shape[0]):
        action.append(np.random.uniform(action_space.low[i], action_space.high[i]))
    return np.asarray(action)


def calc_torque(state, k = 5.0, b = 0.5, k_g = 10.0):
    # state: cos(q), sin(q), dq
    theta = np.arctan2(state[1], state[0])
    # l_m = 1.0
    # theta_0 = np.arcsin(4.0 / k_g)
    # energy_error = 0.5 * k_g * (np.cos(theta_0) - np.cos(theta)) - \
    #                k_g / (9.81 * l_m) * l_m ** 2.0 * state[-1] ** 2.0 / 3.0
    # if energy_error >= 0:
    #     torque = np.abs([k * (0 - theta) - b * state[-1]]) * np.sign(state[-1])
    # else:
    #     torque = np.asarray([k * (0 - theta) - b * state[-1]]) - k_g * state[1]
    torque = np.asarray([k * (0 - theta) - b * state[-1]]) - k_g * state[1]
    # print('Set torque: ', torque)
    return torque, theta


def render_env(env, agent, k = None, b = None, k_g = None,
               model_based = False, save_video = False):
    state = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        if k is None or b is None or k_g is None:
            action = agent.select_action(state, eval=True)
            if model_based:
                # torque, theta = calc_torque(state, k=abs(action[0]), b=0.5, k_g=20.0)
                torque, theta = calc_torque(state, k=action[0], b=action[1], k_g=20.0)
            else:
                torque = action
        else:
            torque, theta = calc_torque(state, k, b, k_g)
        next_state, reward, done, _ = env.step(torque)
        episode_reward += reward
        state = next_state
        if save_video:
            env.render(mode='rgb_array')
        else:
            env.render()
    print('Render reward: ', episode_reward)

def eval_agent(env, agent, model_based = False, episodes = 10):
    avg_reward = 0.

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, eval=True)
            if model_based:
                # torque, theta = calc_torque(state, k=abs(action[0]), b=0.5, k_g=20.0)
                torque, theta = calc_torque(state, k=action[0], b=action[1], k_g=20.0)
            else:
                torque = action
            next_state, reward, done, _ = env.step(torque)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes
    return avg_reward
