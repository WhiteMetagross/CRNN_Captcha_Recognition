cmake_minimum_required(VERSION 3.16)
project(CaptchaInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(inference inference.cpp)

target_link_libraries(inference PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)