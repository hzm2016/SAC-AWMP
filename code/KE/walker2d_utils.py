import math
import torch
import numpy as np


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


def calc_torque(state, action):
    # state: cos(q), sin(q), dq
    if action is None:
        k = np.ones(6)
        b = 0.1 * np.ones(6)
        joint_angle_e = np.ones(6)
    else:
        k = action[0::3]
        b = action[1::3]
        joint_angle_e = action[2::3]
    joint_angle = state[8:20:2]
    joint_speed = state[9:20:2]
    torque = k * (joint_angle_e - joint_angle) - b * joint_speed
    return torque, joint_angle


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
                torque, theta = calc_torque(state, action)
            else:
                torque = action
        else:
            torque, theta = calc_torque(state, action)
        next_state, reward, done, _ = env.step(torque)
        xyz = state[0:3]
        pitch = state[7]
        if abs(xyz[2]) > 0.8 or abs(pitch) > 1.0:
            my_reward = -2.0
            done = True
        episode_reward += reward
        state = next_state
        if save_video:
            env.render(mode='rgb_array')
        else:
            env.render()
    print('Render reward: ', episode_reward)


def render_env_phase(env, agent, save_video = False):
    state = env.reset()
    action = agent.select_action(state, eval=True)
    pre_state = np.copy(state)
    done = False
    episode_reward = 0.0
    while not done:
        is_same_phase = np.array_equal(pre_state[-2:], state[-2:])
        if not is_same_phase:
            action = agent.select_action(state, eval=True)

        torque, theta = calc_torque(state, action)
        next_state, reward, done, _ = env.step(torque)

        episode_reward += reward

        state[:] = next_state[:]
        pre_state[:] = state[:]

        if save_video:
            env.render(mode='rgb_array')
            print('action: ', action)
        else:
            env.render()
    print('Render reward: ', episode_reward)


def eval_agent_phase(env, agent, episodes = 10):
    avg_reward = 0.

    for _ in range(episodes):
        state = env.reset()
        # Sample action from policy
        action = agent.select_action(state, eval=True)
        pre_state = np.copy(state)
        episode_reward = 0
        done = False
        while not done:
            is_same_phase = np.array_equal(pre_state[-2:], state[-2:])
            if not is_same_phase:
                action = agent.select_action(state, eval=True)
            torque, theta = calc_torque(state, action)
            next_state, reward, done, _ = env.step(torque)
            episode_reward += reward

            state[:] = next_state[:]
            pre_state[:] = state[:]

        avg_reward += episode_reward
    avg_reward /= episodes
    return avg_reward


def eval_agent(env, agent, model_based = False, episodes = 10):
    avg_reward = 0.
    ave_x_dist = 0.0
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, eval=True)
            if model_based:
                # torque, theta = calc_torque(state, k=abs(action[0]), b=0.5, k_g=20.0)
                torque, theta = calc_torque(state, action)
            else:
                torque = action
            next_state, reward, done, _ = env.step(torque)
            xyz = state[0:3]
            pitch = state[7]
            if abs(xyz[2]) > 0.8 or abs(pitch) > 1.0:
                done = True
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
        ave_x_dist += state[0]
    avg_reward /= episodes
    ave_x_dist /= episodes
    return avg_reward, ave_x_dist
