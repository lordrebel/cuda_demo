CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
#设置cuda cc
set(CMAKE_CUDA_ARCHITECTURES 80)
#使能cuda
project(cuda_demo LANGUAGES CXX CUDA)
if(CUDA_ENABLED)
    enable_language(CUDA)
endif()
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

