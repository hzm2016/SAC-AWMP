from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
from mpc.env_dx import leg
from IPython.display import HTML
from tqdm import tqdm

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
import random

def uniform(shape, low, high):
    r = high - low
    return torch.rand(shape) * r + low


def plot_mpc(theta_obj, theta, theta_mpc, u, u_mpc, t, mpc_T):
    # plotting for forced prediction
    plt.clf()
    t_pre = np.arange(0, t+1)
    t_after = np.arange(t+1,t+1+mpc_T)
    # print(theta_obj.shape, theta.shape, theta_mpc.shape, u.shape,
    #       u_mpc.shape, t_pre.shape, t_after.shape)

    plt.plot(t_pre, theta_obj[0:t + 1], 'r-', linewidth=2, label='Objective')
    plt.plot(t_after, theta_obj[t] * np.ones(t_after.shape), 'r--', linewidth=2)
    plt.plot(t_pre, theta[0:t + 1], 'k-', linewidth=2, label='States')
    plt.plot(t_after, theta_mpc, 'k--', linewidth=2, label='Predicted states')
    # plt.step(t_pre, 1e-3 * u[0:t + 1], 'b--', linewidth=2, label='Input')
    # plt.step(t_after, u_mpc, 'b--', linewidth=2)
    plt.axvline(x=t)
    plt.axis([0, T + mpc_T, -0.6, 0.6])
    plt.xlabel('time', fontsize=16)
    plt.ylabel('y(t)', fontsize=16)
    plt.draw()
    plt.legend()
    plt.pause(0.01)


def fifo_list(val_list, val):
    val_list[:-1] = val_list[1:]
    val_list[-1] = val
    return val_list


def calc_objective(goal_state = torch.Tensor((1., 0., 0.)),
                   mode = 'swingup', ctrl_penalty = 1e-6):
    if mode == 'swingup':
        # w_cos_theta, w_sin_theta, w_d_theta
        goal_weights = torch.Tensor((1., 1., 1e-6))
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(dx.n_ctrl)
        ))
        px = -torch.sqrt(goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(dx.n_ctrl)))
        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
            mpc_T, n_batch, 1, 1
        )
        p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)
    elif mode == 'spin':
        Q = 0.001 * torch.eye(dx.n_state + dx.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(
            mpc_T, n_batch, 1, 1
        )
        p = torch.tensor((0., 0., -1., 0.))
        p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)
    return Q, p


def calc_nln_zmp(x, u, dx):
    cos_angle = x[0]
    sin_angle = x[1]
    angle_v = x[2]
    g, m, l = dx.params.detach().numpy()
    angle_a = -3. * g / (2. * l) * (-sin_angle) + 3. * u / (m * l ** 2)

    x_g = - 0.5 * l * sin_angle
    x_a = 0.5 * l * sin_angle * angle_v ** 2 - 0.5 * l * cos_angle * angle_a
    h = 0.5 * l * cos_angle
    return x_g - h * x_a / g


# gravity (g), mass (m), length (l)
params = torch.tensor((9.81, 5., 0.5))
# LegDx is a neural network module
dx = leg.LegDx(params, simple=True)

n_batch, T, mpc_T = 1, 1000, 20
is_save_video = False
is_visualize_leg = True

torch.manual_seed(0)
angle = uniform(n_batch, 0.0, 0.0)
angle_v = uniform(n_batch, 0.0, 0.0)
x_init = torch.stack((torch.cos(angle), torch.sin(angle), angle_v), dim=1)

x = x_init
u_init = None

t_dir = tempfile.mkdtemp()
print('Tmp dir: {}'.format(t_dir))

if is_visualize_leg:
    n_row, n_col = 1, 2
    fig, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))
    axs = axs.reshape(-1)

#  Create plot
plt.figure(figsize=(10,4))
plt.ion()
plt.show()

theta_obj = np.zeros(0)
theta = np.zeros(0)
u = np.zeros(0)
obj_noise = [0, 0]
for t in tqdm(range(T)):
    start = time.time()
    # nominal_states size: (mpc_T, n_batch, n_states)
    # if random.uniform(0, 1.0) > 0.8 and t % 10 == 0:
    #     obj_noise = fifo_list(obj_noise, random.uniform(-0.1, 0.1) + 0.5 * np.sin(t / 2.0))
    theta_obj = np.append(theta_obj, obj_noise[-1])
    Q, p = calc_objective(goal_state=torch.Tensor((np.cos(theta_obj[-1]), np.sin(theta_obj[-1]), 0.)))
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        dx.n_state, dx.n_ctrl, mpc_T,
        u_init=u_init,
        # u_lower=dx.lower, u_upper=dx.upper,
        lqr_iter=5,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(x, QuadCost(Q, p), dx)
    # x size: (n_batch, n_states)
    # u_init size: (mpc_T, n_batch, n_states)

    # print(nominal_states.shape, nominal_actions.shape, nominal_objs.shape)

    theta = np.append(theta, np.arctan2(x[0, 1].detach().numpy(), x[0, 0].detach().numpy()))
    u = np.append(u, nominal_actions[0, 0, 0].detach().numpy())
    print('time: ', time.time() - start ,', x_zmp: ', calc_nln_zmp(x[0].detach().numpy(), u[-1], dx))
    # if t % 10 == 0:
    plot_mpc(theta_obj, theta, np.arctan2(nominal_states[:, 0, 1].detach().numpy(),
                                          nominal_states[:, 0, 0].detach().numpy(), ), u,
             nominal_actions[:, 0, 0].detach().numpy(), t, mpc_T)

    next_action = nominal_actions[0]
    # LegDx is a neural network module, and this is a forward function

    if random.uniform(0, 1.0) > 0.9:
        next_action += random.uniform(-5.0, 5.0) + 1.0 * np.sin(t / 30.0)

    x = dx(x, next_action)
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]

    if is_visualize_leg:
        for i in range(n_batch):
            dx.get_frame(x[i], ax=axs[i])
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        fig.tight_layout()
    # fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
    # plt.close(fig)
# %%

if is_save_video:
    mode = 'swingup'
    vid_fname = 'pendulum-{}.mp4'.format(mode)

    if os.path.exists(vid_fname):
        os.remove(vid_fname)

    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        t_dir, vid_fname)
    os.system(cmd)
    print('Saving video to: {}'.format(vid_fname))
    # %%
    video = io.open(vid_fname, 'r+b').read()
    encoded = base64.b64encode(video)
    HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))