cmake_minimum_required(VERSION 3.15)
project(CV_Project_2)

# Set the C++ standard you wish to use (you could use 11 or 14 or 17 depending on your needs)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package in your system
find_package(OpenCV REQUIRED)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable for your project, compiling main.cpp, functions.cpp, and csv_util.cpp into it
add_executable(main src/main.cpp src/functions.cpp src)


# Link your executable with the OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})
