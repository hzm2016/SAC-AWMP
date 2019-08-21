"""Sample Webots controller for the inverted pendulum benchmark."""

from utils import *
import time

controller = Controller_timer()
controller.start()
time.sleep(50)
controller.stop()


# # Get pointer to the robot.
# robot = Robot()
# sampling_rate = 100
# # Get the time step of the current world.
# time_step = int(robot.getBasicTimeStep())
# ankle_z_motor = robot.getMotor('ankle z rotational motor')
# ankle_z_position = robot.getPositionSensor('ankle z position sensor')
# ankle_x_linear_motor = robot.getMotor('ankle x linear motor')
# ankle_y_linear_motor = robot.getMotor('ankle y linear motor')
# ankle_z_linear_motor = robot.getMotor('ankle z linear motor')
# foot_force_sensor = robot.getTouchSensor('foot touch sensor')
#
# ankle_z_motor.enableTorqueFeedback(sampling_rate)
# ankle_z_position.enable(sampling_rate)
# ankle_x_linear_motor.enableForceFeedback(sampling_rate)
# ankle_y_linear_motor.enableForceFeedback(sampling_rate)
# ankle_z_linear_motor.enableForceFeedback(sampling_rate)
# foot_force_sensor.enable(sampling_rate)
#
# theta_e = 0
# k = 5.00
# b = k/2.0
# g = 9.81
# theta_list = [0, 0]
# time_list = [0, 0]
#
#
# robot_thread = Robot_timer(robot, ankle_z_motor, ankle_z_position, time_step)
# new_paras = pyqtSignal(float, float, float)
# new_paras.connect(robot_thread.update_paras)
# robot_thread.start()
# theta_e_vec = [0, 0.1]
# for i in range(10):
#     theta_e = theta_e_vec[i % 2]
#     new_paras.emit(k, b, theta_e)
#     time.sleep(5)
#
# robot_thread.stop()

# while robot.step(time_step) != -1:
#     tau_z = ankle_z_motor.getTorqueFeedback()
#     f_x = ankle_x_linear_motor.getForceFeedback()
#     f_y = ankle_y_linear_motor.getForceFeedback()
#     f_z = ankle_z_linear_motor.getForceFeedback()
#     print('tau_z: %s, f_x: %s, f_y: %s, f_z: %s' % (tau_z, f_x, f_y, f_z))
#     print('foot force sensor getValues: ', foot_force_sensor.getValues())
#     theta_list = fifo_list(theta_list, ankle_z_position.getValue())
#     time_list = fifo_list(time_list, robot.getTime())
#     if 0 == time_list[1] - time_list[0]:
#         continue
#     angle_z_v = (theta_list[1] - theta_list[0]) / (time_list[1] - time_list[0])
#     print('angle: %s, velocity: %s' % (theta_list[-1], angle_z_v))
#     tau = k * (theta_e - theta_list[-1]) - b * angle_z_v - 5 * g * 0.4 * math.sin(theta_list[-1])
#     # tau = k * (theta_e - theta_list[-1]) - b * angle_z_v
#     # if tau < 1e-3:
#     #     tau = 0
#     if time_list[-1] < 1:
#         print('time: %s' % time_list[-1])
#         ankle_z_motor.setPosition(0.1)
#         continue
#     # tau = 3.0 * math.sin(5 * time)
#     ankle_z_motor.setTorque(tau)
#     print('tau: %s' % tau)

