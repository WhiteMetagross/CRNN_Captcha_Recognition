cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aster_crnn_ctc_stn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(onnxruntime REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)

add_executable(captcha_solver
    src/main.cpp
    src/model.cpp
    src/data.cpp
    src/util.cpp
)

target_link_libraries(captcha_solver
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    onnxruntime
)

set_property(TARGET captcha_solver PROPERTY CXX_STANDARD 17)