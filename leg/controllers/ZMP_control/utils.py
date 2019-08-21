from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from controller import Robot
import math
import time
import random
import numpy as np

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
            angle_vec = fifo_list(angle_vec, self.ankle_z_position.getValue())
            angle_v = (angle_vec[-1] - angle_vec[-2]) / (self.time_step * 1e-3)
            angle_v_vec = fifo_list(angle_v_vec, angle_v)
            angle_a = (angle_v_vec[-1] - angle_v_vec[-2]) / (self.time_step * 1e-3)
            time_vec = fifo_list(time_vec, self.robot.getTime())
            if time_vec[-1] < 1.0:
                # print('robot time: %s, real time: %s' % (time_list[-1], time.time() - start))
                self.ankle_z_motor.setPosition(-0.1)
                continue
            tau = self.k * (self.angle_e - angle_vec[-1]) - self.b * angle_v - \
                  5.0 * g * 0.4 * math.sin(angle_vec[-1])
            if random.uniform(0, 1.0) > 0.8:
                tau_noise = random.uniform(2, 5.0) + 5 * math.sin(3.0 * time_vec[-1])
            else:
                tau_noise = 0.0
            self.ankle_z_motor.setTorque(tau + tau_noise)
            x_zmp_vec = fifo_list(x_zmp_vec, calc_x_zmp_by_mc(angle_vec[1], angle_v, angle_a))
            # x_zmp = x_zmp_vec[np.argmin(np.abs(x_zmp_vec))]
            x_zmp = x_zmp_vec[-1]
            x_zmp_th = 0.03
            # if abs(x_zmp) < x_zmp_th:
            #     self.k = 10.0
            #     self.b = self.k / 2.0
            # elif (abs(x_zmp) < x_zmp_th * 1.5) and abs(x_zmp_vec[-1]) < abs(x_zmp_vec[-2]):
            # # if (abs(x_zmp) > x_zmp_th / 2.0) and (abs(x_zmp) < x_zmp_th * 1.5) and \
            # #         abs(x_zmp_vec[-1]) < abs(x_zmp_vec[-2]):
            #     self.k *= x_zmp_th / abs(x_zmp)
            #     self.b = self.k / 2.0
            #     print('x_zmp: %s, k: %s, b: %s.' % (x_zmp, self.k, self.b))
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

    def run(self):
        self.is_run = True
        self.robot_thread.start()
        KP = 0.05
        KI = 0.0
        KD = KP / 2.0
        integral = 0.0
        previous_error = 0.0
        x_zmp_origin = 0.0
        time_list = [0.0, 0.0]
        angle_e_list = [0.0, 0.0]
        while self.is_run:
            # time_list = fifo_list(time_list, self.robot.getTime())
            # if time_list[-1] < 0.5:
            #     print('control time: ', time_list[-1])
            #     time.sleep(1.0 / self.sampling_rate)
            #     continue
            # if 0 == time_list[1] - time_list[0]:
            #     time.sleep(1.0 / self.sampling_rate)
            #     continue
            # F_x = self.ankle_x_linear_motor.getForceFeedback()
            # F_y = self.ankle_y_linear_motor.getForceFeedback()
            # print('x_zmp: %s, F_x: %s, Fy: %s' % (self.x_zmp, F_x, F_y))
            # x_zmp = calc_x_zmp_by_mc(self.angle, self.angle_v, self.angle_a)
            # print('x_zmp: ', x_zmp)
            # self.angle_e = angle_e_list[int(time_list[-1] / 5) % 2]
            # F_x = self.ankle_x_linear_motor.getForceFeedback()
            # F_y = self.ankle_y_linear_motor.getForceFeedback()
            # x_zmp = calc_x_zmp(F_x=F_x, F_y= F_y, tau_m=self.tau)
            # x_zmp_error = x_zmp_origin - x_zmp
            # print('x_zmp: ', x_zmp)
            # # PID control.
            # integral = integral + (x_zmp_error + previous_error) * 0.5 * (time_list[1] - time_list[0])
            # derivative = (x_zmp_error - previous_error) / (time_list[1] - time_list[0])
            # x_zmp_offset = KP * x_zmp_error + KI * integral + KD * derivative
            # tau_design = calc_tau_m(F_x=F_x, F_y=F_y, x_zmp=x_zmp_origin + x_zmp_offset)
            # self.k_a = calc_k_a(tau_design, self.k, self.b, self.angle_e, self.angle, self.angle_v)
            # self.update_impedance()
            time.sleep(1.0 / self.sampling_rate)
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
        return -r * math.sin(angle) + (r * angle_v**2 * math.sin(angle) + r*angle_a*math.cos(angle)) / g_offset

def calc_tau_m(F_x, F_y, x_zmp, m_f = 0.5, x_g = 0.025, h_ankle = 0.05):
    return (m_f * 9.81 + F_y) * x_zmp - F_x * h_ankle - m_f * 9.81 * x_g


def calc_k_a(tau_design, k, b, angle_e, angle, angle_v, m = 5.0, l = 0.4):
    g = 9.81
    tau_design = tau_design + m * g * l * math.sin(angle)
    tau_old = k * (angle_e - angle) - b * angle_v
    print('tau_old: %s, tau_design: %s.' % (tau_old, tau_design))
    if abs(tau_old) < 1e-3:
        return 1.0
    else:
        return 1.0
        # return abs(tau_design / tau_old)


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
