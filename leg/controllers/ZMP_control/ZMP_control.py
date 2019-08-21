"""Sample Webots controller for the inverted pendulum benchmark."""

from utils import *
import time

controller = Controller_timer()
controller.start()
time.sleep(50)
controller.stop()