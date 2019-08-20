#include <iostream>
#include "webots_cpp/Robot.hpp"
#include "webots_cpp/Motor.hpp"

int main() {
    webots::Robot robot;
    int timeStep = (robot.getBasicTimeStep());
    Motor *pitchMotor = robot.getMotor("pitch motor");
    pitchMotor->setPosition(1.0);
    std::cout << "Hello, World!" << std::endl;
    return 0;

}