"""

Inverted Pendulum MPC control

author: Atsushi Sakai

"""
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random


l = 0.4  # length of bar
m_foot = 0.5  # [kg]
m_body = 5.0 # [kg]
g = 9.81  # [m/s^2]

Q = np.diag([1.0, 0.0])
R = np.diag([0.00])
nx = 2   # number of state
nu = 1   # number of input
T = 10  # Horizon length
delta_t = 0.1  # time tick

animation = True

def main():
    x0 = np.array([
        [0.1],
        [0.1]
    ])

    x = np.copy(x0)

    itr_num = 50
    theta_vec = []
    for i in range(itr_num):
        # try:
        otheta, dtheta, ou = mpc_control(x)
        # except:
        #     otheta = x[0, 0]
        #     dtheta = x[1, 0]
        #     ou = [5.0 * (0.0 - theta) - 2.5 * dtheta - m_body * g * l * math.sin(theta)]
        u = ou[0]
        if random.uniform(0, 1.0) > 0.8:
            u_noise = random.uniform(2.0, 5.0) + 5.0 * math.sin(i / 3.0)
        else:
            u_noise = 0.0
        u = u + u_noise
        x = simulation(x, u)

        if animation:
            plt.clf()
            theta_vec.append(float(x[0]))
            plt.plot(np.asarray(theta_vec))

            # show_leg(theta)
            plt.xlim([0, itr_num + 5])
            # plt.ylim([-5.0, 5.0])
            plt.pause(0.001)


def simulation(x, u):

    A, B = get_model_matrix()

    x = np.dot(A, x) + np.dot(B, u)

    return x

# def calc_x_zmp(x, u):
#     x_zmp = cvxpy.abs(- l * x[0] + u[0] / (m_body * g))
#     return x_zmp

def calc_cost(xu):
    xu = xu.reshape((nx + nu, T + 1))
    cost = 0.0
    Q = np.diag([1.0, 0.0])
    R = np.diag([0.00])
    x = xu[:-1, :]
    u = xu[-1, :].reshape((1, -1))
    for t in range(T):
        cost += np.dot(np.dot(x[:, t + 1].reshape((1, -1)), Q),
                       x[:, t + 1])
        cost += np.dot(np.dot(u[:, t].reshape((1, -1)), R), u[:, t])
    return cost

def state_constraints(xu):
    xu = xu.reshape((nx + nu, T + 1))
    x = xu[:-1, :]
    u = xu[-1, :].reshape((1, -1))
    A, B = get_model_matrix()
    constr = np.zeros(T)
    for t in range(T):
        error = np.dot(A, x[:, t]) + np.dot(B, u[:, t]) - x[:, t + 1]
        constr[t] = np.dot(error.reshape((1, -1)), error)
    return constr

# def state_constraint(x_end_2, u_end, A, B):
#     return A * x_end_2[:, 0] + B * u_end - x_end_2[:, 1]
#
# def init_constraint(x_init, x_0):
#     return x_init - x_0


def zmp_constraints(xu):
    xu = xu.reshape((nx + nu, T + 1))
    x = xu[:-1, :]
    u = xu[-1, :].reshape((1, -1))
    constr = np.zeros(T)
    for t in range(T):
        constr[t] = 0.5 - calc_nln_zmp(x[t], u[t])
    return constr


def calc_nln_zmp(x, u, g = 9.81):
    x_g = -l * math.sin(x[0])
    angle_a = u[0] / (m_body * l ** 2)
    x_a = l * math.sin(x[0]) * x[1] ** 2 - l * math.cos(x[0]) * angle_a
    h = l * math.cos(x[0])
    return x_g - h * x_a / g


def mpc_control(x0):
    xu = np.zeros((nx + nu, T + 1))
    xu[:nx, :] += x0.reshape((-1, 1))
    xu = xu.reshape((nx + nu) * (T + 1))
    # constr1 = {'type': 'ineq', 'fun': zmp_constraints}
    constr2 = {'type': 'eq', 'fun': state_constraints}
    constr = ([constr2])
    start = time.time()
    solution = minimize(calc_cost, xu, method='SLSQP',
                        constraints=constr)
    xu = solution.x
    xu = xu.reshape((nx + nu, T + 1))
    elapsed_time = time.time() - start
    print("calc time:{0} [sec]".format(elapsed_time))
    print('x_u.shape: ', xu.shape)
    theta = xu[0, :]
    dtheta = xu[1, :]
    ou = xu[-1, :-1]
    print('zmp: ', calc_nln_zmp([theta[0], dtheta[0]], [ou[0]]))
    return theta, dtheta, ou


def get_nparray_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()


def get_model_matrix():
    # Model Parameter
    A = np.array([
        [0.0, 1.0],
        [g / (l * delta_t), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / (m_foot * l **2)]
    ])
    B = delta_t * B

    return A, B


def flatten(a):
    return np.array(a).flatten()


def show_leg(theta):
    xt = 0.0
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.0

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l * math.sin(-theta)])
    bx += xt
    by = np.matrix([cart_h, l * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")


def visualize_test():

    #  x = 1.0
    #  theta = math.radians(10.0)
    #  show_cart(x, theta)
    #  plt.show()

    angles = np.arange(-math.pi / 2.0, math.pi / 2.0, math.radians(1.0))

    xl = [2.0 * math.cos(i) for i in angles]

    for x, theta in zip(xl, angles):
        plt.clf()
        show_leg(x, theta)
        plt.pause(0.001)


if __name__ == '__main__':
    main()
    #  visualize_test()
