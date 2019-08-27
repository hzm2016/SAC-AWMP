from tqdm import tqdm
from mpc import mpc
from mpc import QuadCost, LinDx
import matplotlib.pyplot as plt
import torch


def plot_mpc(x, u):
    # plotting for forced prediction
    plt.clf()
    plt.plot(x,'r-',linewidth=2, label='State')
    # plt.plot(obj,'k-',linewidth=2,label='Objective')
    plt.plot(u,'b--',linewidth=2, label='Input')
    # plt.axvline(x=i)
    # plt.axis([0, ns+P, -15, 15])
    plt.xlabel('Frame',fontsize=16)
    plt.ylabel('x(t)',fontsize=16)
    plt.draw()
    plt.legend()
    plt.show()
    # plt.pause(0.1)

# def forward(x, u):


torch.manual_seed(0)

n_batch, n_state, n_ctrl, T = 10, 2, 1, 5
n_sc = n_state + n_ctrl

# Randomly initialize a PSD quadratic cost and linear dynamics.
C = torch.randn(T*n_batch, n_sc, n_sc)
C = torch.bmm(C, C.transpose(1, 2)).view(T, n_batch, n_sc, n_sc)
c = torch.randn(T, n_batch, n_sc)

alpha = 0.2
delta_t = 0.1
ml2 = 0.8
A = torch.eye(n_state)
A[0, 1] = delta_t
B = torch.zeros(n_state, n_ctrl)
B[-1, 0] = delta_t / ml2

F = torch.cat((A, B), dim=1).repeat(T, n_batch, 1, 1)

# R = A.repeat(T, n_batch, 1, 1)
#
# S = torch.randn(T, n_batch, n_state, n_ctrl)
# F = torch.cat((R, S), dim=3)

# The initial state.
x_init = torch.randn(n_batch, n_state)

# The upper and lower control bounds.
u_lower = -10.0 * torch.ones(T, n_batch, n_ctrl)
u_upper = 10.0 * torch.ones(T, n_batch, n_ctrl)
x = x_init

for t in tqdm(range(T)):
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        n_state, n_ctrl, T,
        u_init=u_init,
        u_lower=u_lower, u_upper=u_upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        # linesearch_decay=dx.linesearch_decay,
        # max_linesearch_iter=dx.max_linesearch_iter,
        # grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(x, QuadCost(C, c), LinDx(F))

    next_action = nominal_actions[0]
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]

    # x = dx(x, next_action)
    plot_mpc(nominal_states.numpy()[0, :, 0], nominal_actions.numpy()[0, :, 0])

    # n_row, n_col = 4, 4
    # fig, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))
    # axs = axs.reshape(-1)
    # for i in range(n_batch):
    #     dx.get_frame(x[i], ax=axs[i])
    #     axs[i].get_xaxis().set_visible(False)
    #     axs[i].get_yaxis().set_visible(False)
    # fig.tight_layout()
    # fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
    # plt.close(fig)
# %%


nominal_states, nominal_actions, nominal_objs = mpc.MPC(
    n_state=n_state,
    n_ctrl=n_ctrl,
    T=T,
    u_lower=u_lower,
    u_upper=u_upper,
    lqr_iter=20,
    verbose=1,
    backprop=True,
    exit_unconverged=False,
)(x_init, QuadCost(C, c), LinDx(F))

print(nominal_states.shape, nominal_actions.shape, nominal_objs.shape)
plot_mpc(nominal_states.numpy()[0, :, 0], nominal_actions.numpy()[0, :, 0])
