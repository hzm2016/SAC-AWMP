import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2

class Figure_2D():
    def __init__(self, width=10, height=7.5, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        plt.rcParams.update({'font.size': 12, 'font.serif': 'Times New Roman'})
        self.axes = self.fig.add_subplot(111)

    def update_plot(self, y_vec, legend_vec=None, block = False):
        if legend_vec is None:
            legend_vec = ['True Q', 'TD3 y', 'TD3 Q_a', 'TD3 Q_b',
                          'ATD3 y', 'ATD3 Q_a', 'ATD3 Q_b', 'ATD3 Q_m']
        self.axes.clear()
        self.line_list = self.axes.plot(y_vec)
        self.axes.legend(legend_vec, loc='lower center', ncol=4, bbox_to_anchor=(0.50, 1.0), frameon=False)
        self.axes.figure.canvas.draw_idle()
        plt.show(block=block)
        plt.pause(0.0001)
        plt.ylabel('Q-value')
        plt.xlabel('Time steps')


def fig_to_img(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

save_video = True
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('Q.mp4', fourcc, 100.0, (1000, 750))

gamma = 0.99
tau = 0.1
t_freq = 2
alpha = 0.05
beta = 0.2
step_num = 10
Q_a_ATD3 = np.random.uniform(0.0, 2 * step_num, step_num)
Q_a_TD3 = np.copy(Q_a_ATD3)
Q_t_a_ATD3 = np.copy(Q_a_ATD3)
Q_t_a_TD3 = np.copy(Q_a_ATD3)
Q_b_ATD3 = np.random.uniform(0.0, 2 * step_num, step_num)
Q_b_TD3 = np.copy(Q_b_ATD3)
Q_t_b_ATD3 = np.copy(Q_b_ATD3)
Q_t_b_TD3 = np.copy(Q_b_ATD3)

Q_m_ATD3 = 0.5 * (Q_a_ATD3 + Q_b_ATD3)

Q_true = np.zeros(step_num)
for i in range(step_num):
    Q_true_temp = 0.0
    for t in range(i, step_num):
        Q_true_temp += gamma**(t-i)
    Q_true[i] = Q_true_temp

fig_2d = Figure_2D()
y_vec = np.asarray([Q_true[0], 1.0 + min(Q_a_TD3[1], Q_b_TD3[1]), Q_a_TD3[0], Q_b_TD3[0],
                    1.0 + min(Q_a_ATD3[1], Q_b_ATD3[1]), Q_a_ATD3[0],
                    Q_b_ATD3[0], Q_m_ATD3[0]]).reshape((1, -1))
fig_2d.update_plot(y_vec=y_vec)
batch_size = 100
for r in range(1000):
    y_TD3 = np.ones(step_num)
    y_TD3[:(step_num-1)] = np.ones(step_num-1) + gamma * np.min(np.c_[Q_t_a_TD3[1:], Q_t_b_TD3[1:]], axis=-1)
    y_ATD3 = np.ones(step_num)
    y_ATD3[:(step_num-1)] = np.ones(step_num-1) + gamma * np.min(np.c_[Q_t_a_ATD3[1:], Q_t_b_ATD3[1:]], axis=-1)

    Q_a_TD3 = Q_a_TD3 + alpha * (y_TD3 - Q_a_TD3)
    Q_b_TD3 = Q_b_TD3 + alpha * (y_TD3 - Q_b_TD3)

    d_ab = Q_a_ATD3 - Q_b_ATD3
    Q_a_ATD3 = Q_a_ATD3 + alpha * (y_ATD3 - Q_a_ATD3 + beta * d_ab)
    Q_b_ATD3 = Q_b_ATD3+ alpha * (y_ATD3 - Q_b_ATD3 - beta * d_ab)
    Q_m_ATD3 = 0.5 * (Q_a_ATD3 + Q_b_ATD3)
    if 0 == r % t_freq:
        Q_t_a_TD3 = (1 - tau) * Q_t_a_TD3 + tau * Q_a_TD3
        Q_t_b_TD3 = (1 - tau) * Q_t_b_TD3 + tau * Q_b_TD3

        Q_t_a_ATD3 = (1 - tau) * Q_t_a_ATD3 + tau * Q_a_ATD3
        Q_t_b_ATD3 = (1 - tau) * Q_t_b_ATD3 + tau * Q_b_ATD3

    print('e_a_TD3: {}, e_m_ATD3: {}'.format(Q_a_TD3[0] - Q_true[0], Q_m_ATD3[0] - Q_true[0]))
    # print('Q_a_TD3: {}, Q_m_ATD3: {}'.format(Q_a_TD3, Q_m_ATD3))
    y_vec = np.r_[y_vec,
                  np.asarray([Q_true[0], y_TD3[0], Q_a_TD3[0], Q_b_TD3[0],
                              y_ATD3[0], Q_a_ATD3[0], Q_b_ATD3[0], Q_m_ATD3[0]]).reshape((1, -1))]
    fig_2d.update_plot(y_vec)
    if save_video:
        img = fig_to_img(fig_2d.fig)
        out_video.write(img)
        # cv2.imshow('Q', img)

if save_video:
    out_video.release()

fig_2d.update_plot(y_vec, block = True)


