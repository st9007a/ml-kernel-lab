cmake_minimum_required(VERSION 3.24)

project(ml_kernel_lab_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(show_gpu_info csrc/tools/show_gpu_info.cu)

target_compile_features(show_gpu_info PRIVATE cxx_std_17 cuda_std_17)
