/*
 * Copyright 1996-2019 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
   Description:   Demo for InertialUnit node
*/

#include <stdio.h>
#include <stdlib.h>
#include <webots_c/inertial_unit.h>
#include <webots_c/motor.h>
#include <webots_c/robot.h>


int main(int argc, const char *argv[]) {
  // initialize webots API
  wb_robot_init();

  int step = wb_robot_get_basic_time_step();

  WbDeviceTag pitch_motor = wb_robot_get_device("pitch motor");

  for (int i = 0; i < 1000; i++) {
    // choose a random target
    double time = wb_robot_get_time();
    double pitch = sin(time);

    printf("time #%f: pitch=%f \n", time, pitch);

    wb_motor_set_position(pitch_motor, pitch);
    wb_motor_get_force_feedback(pitch_motor)
    for (int j = 0; true; j++) {
      // execute a simulation step
      if (wb_robot_step(step) == -1)
        break;
    }
  }

  // cleanup webots resources
  wb_robot_cleanup();

  return 0;
}
