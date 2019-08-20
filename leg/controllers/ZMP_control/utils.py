from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from controller import Robot
import math
import time

class Robot_timer(QThread, QObject):
    def __init__(self, robot, ankle_z_motor, ankle_z_position, time_step):
        super(Robot_timer, self).__init__()
        self.robot = robot
        self.ankle_z_motor = ankle_z_motor
        self.ankle_z_position = ankle_z_position
        self.time_step = time_step
        self.k = 5.00
        self.b = self.k / 2.0
        self.theta_e = 0.0
        self.is_run = True

    def update_paras(self, k, b, theta_e):
        print('Received theta_e: ', theta_e)
        self.k = k
        self.b = b
        self.theta_e = theta_e

    def run(self):
        self.is_run = True
        theta_list = [0, 0]
        time_list = [0, 0]
        g = 9.81
        while self.robot.step(self.time_step) != -1 and self.is_run:
            theta_list = fifo_list(theta_list, self.ankle_z_position.getValue())
            time_list = fifo_list(time_list, self.robot.getTime())
            if 0 == time_list[1] - time_list[0]:
                continue
            angle_z_v = (theta_list[1] - theta_list[0]) / (time_list[1] - time_list[0])
            tau = self.k * (self.theta_e - theta_list[-1]) - self.b * angle_z_v - \
                  5.0 * g * 0.4 * math.sin(theta_list[-1])
            if time_list[-1] < 0.5:
                # print('time: %s' % time_list[-1])
                self.ankle_z_motor.setPosition(self.theta_e - 0.1)
                continue
            # print('tau design: ', tau)
            self.ankle_z_motor.setTorque(tau)

    def stop(self):
        self.is_run = False


class Controller_timer(QThread, QObject):
    new_paras = pyqtSignal(float, float, float)
    def __init__(self,):
        super(Controller_timer, self).__init__()
        self.robot = Robot()
        sampling_rate = 100
        # Get the time step of the current world.
        self.time_step = int(self.robot.getBasicTimeStep())
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
        connected = self.new_paras.connect(self.robot_thread.update_paras, type=Qt.DirectConnection)
        print('connected: ', connected)
        self.is_run = True

    def run(self):
        self.is_run = True
        self.robot_thread.start()
        k = 5.00
        b = k / 2.0
        theta_e_vec = [0, 0.1]
        for i in range(10):
            if not self.is_run:
                break
            theta_e = theta_e_vec[i % 2]
            self.new_paras.emit(k, b, theta_e)
            print('Emit theta_e: ', theta_e)
            time.sleep(5)
        self.robot_thread.stop()

    def stop(self):
        self.is_run = False


def fifo_list(val_list, val):
    val_list[:-1] = val_list[1:]
    val_list[-1] = val
    return val_list