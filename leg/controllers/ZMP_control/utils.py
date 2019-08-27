from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from controller import Robot
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
from mpc.env_dx import leg
import math
import random
import numpy as np
import torch
import time


class Robot_timer(QThread, QObject):
    sensor_signal = pyqtSignal(float, float, float, float)
    def __init__(self, robot, ankle_z_motor, ankle_z_position, time_step):
        super(Robot_timer, self).__init__()
        self.robot = robot
        self.ankle_z_motor = ankle_z_motor
        self.ankle_z_position = ankle_z_position
        self.time_step = time_step
        self.k = 10.00
        self.b = self.k / 2.0
        self.angle_e = 0.0
        self.is_run = True

    def update_impedance(self, k, b, angle_z_e):
        self.k = k
        self.b = b
        self.angle_e = angle_z_e
        print('Received impedance, k: %s, b: %s, angle_e: %s.' % (self.k, self.b, self.angle_e))

    def run(self):
        self.is_run = True
        angle_vec = np.asarray([0.0, 0.0, 0.0])
        x_zmp_vec = np.asarray([0.0, 0.0, 0.0])
        angle_v_vec = np.asarray([0.0, 0.0])
        time_vec = np.asarray([0.0, 0.0])
        g = 9.81
        # self.robot.step(self.time_step) is a timer
        while self.robot.step(self.time_step) != -1 and self.is_run:
            start = time.time()
            angle_vec = fifo_list(angle_vec, self.ankle_z_position.getValue())
            angle_v = (angle_vec[-1] - angle_vec[-2]) / (self.time_step * 1e-3)
            angle_v_vec = fifo_list(angle_v_vec, angle_v)
            angle_a = (angle_v_vec[-1] - angle_v_vec[-2]) / (self.time_step * 1e-3)
            time_vec = fifo_list(time_vec, self.robot.getTime())
            if time_vec[-1] < 1.0:
                # print('robot time: %s, real time: %s' % (time_list[-1], time.time() - start))
                self.ankle_z_motor.setPosition(-0.1)
                continue

            x_zmp_vec = fifo_list(x_zmp_vec, calc_x_zmp_by_mc(angle_vec[1], angle_v, angle_a))
            x_zmp = x_zmp_vec[np.argmin(np.abs(x_zmp_vec))]

            # x_zmp_th = 0.05
            # self.k = 10.00
            # self.b = self.k / 2.0
            # if abs(x_zmp) > x_zmp_th:
            #     self.k *= x_zmp_th / abs(x_zmp)
            #     self.b = self.k / 2.0
            #     print('x_zmp: %s, k: %s, b: %s.' % (x_zmp, self.k, self.b))

            tau = self.k * (self.angle_e - angle_vec[-1]) - self.b * angle_v - \
                  5.0 * g * 0.4 * math.sin(angle_vec[-1])

            if random.uniform(0, 1.0) > 1.0:
                tau_noise = random.uniform(-3.0, 3.0) + 10 * math.sin(3.0 * time_vec[-1])
            else:
                tau_noise = 0.0
            self.ankle_z_motor.setTorque(tau + tau_noise)
            self.sensor_signal.emit(angle_vec[1], angle_v, angle_a, x_zmp)

    def stop(self):
        self.is_run = False


class Controller_timer(QThread, QObject):
    new_impedance = pyqtSignal(float, float, float)
    def __init__(self,):
        super(Controller_timer, self).__init__()
        self.robot = Robot()
        sampling_rate = 100
        self.sampling_rate = 100.0
        # Get the time step of the current world.
        self.time_step = int(self.robot.getBasicTimeStep()) # time for a step, unit: ms
        print('self.time_step: ', self.time_step)
        self.ankle_z_motor = self.robot.getMotor('ankle z rotational motor')
        self.ankle_z_position = self.robot.getPositionSensor('ankle z position sensor')
        self.ankle_x_linear_motor = self.robot.getMotor('ankle x linear motor')
        self.ankle_y_linear_motor = self.robot.getMotor('ankle y linear motor')
        self.ankle_z_linear_motor = self.robot.getMotor('ankle z linear motor')
        self.foot_force_sensor = self.robot.getTouchSensor('foot touch sensor')

        self.ankle_z_motor.enableTorqueFeedback(sampling_rate)
        self.ankle_z_position.enable(sampling_rate)
        self.ankle_x_linear_motor.enableForceFeedback(sampling_rate)
        self.ankle_y_linear_motor.enableForceFeedback(sampling_rate)
        self.ankle_z_linear_motor.enableForceFeedback(sampling_rate)
        self.foot_force_sensor.enable(sampling_rate)

        self.robot_thread = Robot_timer(self.robot, self.ankle_z_motor,
                                        self.ankle_z_position, self.time_step)
        self.new_impedance.connect(self.robot_thread.update_impedance, type=Qt.DirectConnection)
        self.robot_thread.sensor_signal.connect(self.update_sensor_signal, type=Qt.DirectConnection)

        self.angle = 0.0
        self.angle_v = 0.0
        self.angle_a = 0.0
        self.x_zmp = 0.0
        self.tau = 0.0
        self.k = 5.00
        self.b = self.k / 2.0
        self.angle_e = 0.0
        self.k_a = 1.0
        self.is_run = True
        self.mpc_T = 20
        self.Q, self.p = calc_objective(goal_state=torch.Tensor((np.cos(0), np.sin(0), 0.)),mpc_T=self.mpc_T)


    def run(self):
        self.is_run = True
        self.robot_thread.start()
        params = torch.tensor((9.81, 5., 0.4))
        dx = leg.LegDx(params, simple=True)
        u_init = None
        print('Currect GPU: ', torch.cuda.current_device())
        while self.is_run:
            start = time.time()
            x = torch.tensor([[np.cos(self.angle), np.sin(self.angle), self.angle_v]])
            nominal_states, nominal_actions, nominal_objs = mpc.MPC(
                dx.n_state, dx.n_ctrl, self.mpc_T,
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
            )(x, QuadCost(self.Q, self.p), dx)
            u_init = torch.cat((nominal_actions[1:], torch.zeros(1, 1, dx.n_ctrl)), dim=0)
            tau_design = nominal_actions[0, 0, 0].detach().numpy()
            self.k_a = calc_k_a(tau_design, self.k, self.b, self.angle_e, self.angle, self.angle_v)
            self.update_impedance()
            print('Run time: ', time.time() - start)
        self.robot_thread.stop()

    def update_impedance(self):
        self.k *= self.k_a
        self.b *= self.k_a
        self.new_impedance.emit(self.k, self.b, self.angle_e)

    def update_sensor_signal(self, angle, angle_v, angle_a, x_zmp):
        self.angle = angle
        self.angle_v = angle_v
        self.angle_a = angle_a
        self.x_zmp = x_zmp
        # print('Received sensor_signal: ', self.angle, self.angle_v, self.tau)

    def stop(self):
        self.is_run = False


def calc_init_tau(t, tau_max, tau_change_time):
    if t < tau_change_time or t > tau_change_time + 0.1:
        return 0.0
    else:
        return tau_max

def calc_x_zmp(F_x, F_y, tau_m, m_f = 0.5, x_g = 0.025, h_ankle = 0.05):
    print('F_x: %s, F_y: %s, tau_m: %s.' % (F_x, F_y, tau_m))
    x_zmp = (F_x * h_ankle + m_f * 9.81 * x_g + tau_m) / (m_f * 9.81 + F_y)
    return x_zmp


def calc_x_zmp_by_mc(angle, angle_v, angle_a, g=9.81, r = 0.4):
    g_offset = g - (-r * angle_v**2 * math.cos(angle) + r*angle_a*math.sin(angle))
    if g_offset < 1e-3:
        return 0.0
    else:
        x_g = -r * math.sin(angle)
        x_a = - (r * angle_v**2 * math.sin(angle) + r*angle_a*math.cos(angle))
        h = r * math.cos(angle)
        return x_g - h * x_a / g_offset


def calc_tau_m(F_x, F_y, x_zmp, m_f = 0.5, x_g = 0.025, h_ankle = 0.05):
    return (m_f * 9.81 + F_y) * x_zmp - F_x * h_ankle - m_f * 9.81 * x_g


def calc_k_a(tau_design, k, b, angle_e, angle, angle_v, m = 5.0, l = 0.4):
    g = 9.81
    tau_design_offset = tau_design - m * g * l * math.sin(angle)
    tau_old_offset = k * (angle_e - angle) - b * angle_v
    print('tau_old: %s, tau_design: %s.' % (tau_old_offset, tau_design_offset))
    if abs(tau_old_offset) < 1.0:
        return 1.0
    else:
        # return 1.0
        return abs(tau_design_offset / tau_old_offset)


def fifo_list(val_list, val):
    val_list[:-1] = val_list[1:]
    val_list[-1] = val
    return val_list


def pid_control(robot, timestep, leftMotor, rightMotor, ps, maxSpeed = 1000,
                KP = 31.4, KI = 100.5, KD = 0):
    # Define the PID control constants and variables.
    integral = 0.0
    previous_position = 0.0

    # Initialize the robot speed (left wheel, right wheel).
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)

    # Main loop: perform a simulation step until the simulation is over.
    while robot.step(timestep) != -1:
        # Read the sensor measurement.
        position = ps.getValue()

        # Stop the robot when the pendulum falls.
        if math.fabs(position) > math.pi * 0.5:
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            break

        # PID control.
        integral = integral + (position + previous_position) * 0.5 / timestep
        derivative = (position - previous_position) / timestep
        speed = KP * position + KI * integral + KD * derivative

        # Clamp speed to the maximum speed.
        if speed > maxSpeed:
            speed = maxSpeed
        elif speed < -maxSpeed:
            speed = -maxSpeed

        # Set the robot speed (left wheel, right wheel).
        leftMotor.setVelocity(-speed)
        rightMotor.setVelocity(-speed)

        # Store previous position for the next controller step.
        previous_position = position


def calc_objective(n_ctrl = 1, goal_state = torch.Tensor((1., 0., 0.)),
                   ctrl_penalty = 1e-6, mpc_T = 20, n_batch = 1):
    # w_cos_theta, w_sin_theta, w_d_theta
    goal_weights = torch.Tensor((1., 1., 1e-6))
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(n_ctrl)
    ))
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1)
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)
    return Q, p