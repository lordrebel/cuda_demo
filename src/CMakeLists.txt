
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernels)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#将kernel单独编译成lib
add_library(cuda_test SHARED ${CMAKE_CURRENT_SOURCE_DIR}/kernels/test.cu)
add_executable(demo main.cpp)
target_link_libraries(demo cuda_test)