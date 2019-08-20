cmake_minimum_required(VERSION 3.14)
project(Clion_project C)

include_directories(D:/4-code/3rd_parties/Webot)

set(CMAKE_C_STANDARD 99)


add_executable(Clion_project devices/controllers/ZMP_control_c/ZMP_control_c.c devices/controllers/ZMP_control_cpp/ZMP_control_cpp.cpp)