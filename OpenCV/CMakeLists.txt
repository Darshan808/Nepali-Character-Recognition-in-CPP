set(OpenCV_DIR "C:/opencv/build")

cmake_minimum_required(VERSION 3.5)
project(OpenCVProject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVExecutable main.cpp)

target_link_libraries(OpenCVExecutable ${OpenCV_LIBS})
