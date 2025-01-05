cmake_minimum_required(VERSION 3.0)


# Project name
project(HelloOpenCV)


# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(HelloOpenCV
    src/main.cpp
)

target_link_libraries(HelloOpenCV
    ${OpenCV_LIBS}
)

target_include_directories(HelloOpenCV
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)





