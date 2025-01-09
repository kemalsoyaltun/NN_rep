cmake_minimum_required(VERSION 3.10)
project(OpenCVExample)

# Anaconda ortamındaki OpenCV'yi bulmak için bu satırı ekle
set(OpenCV_DIR $ENV{CONDA_PREFIX}/lib/cmake/opencv4)

set(SOURCES
    MNISTread.cpp
    neuralnetw.cpp
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVExample ${SOURCES})
target_link_libraries(OpenCVExample ${OpenCV_LIBS})