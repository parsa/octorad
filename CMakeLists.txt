cmake_minimum_required(VERSION 3.10)
project(octorad CXX CUDA)

option(OCTORAD_WITH_PLAYGROUND "" OFF)

###############################################################################
# Enable C++14
###############################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# Playground
###############################################################################
if(OCTORAD_WITH_PLAYGROUND)
    add_executable(01-heat-conduction playground/01-heat-conduction.cu playground/cuda-gpuinfo.h)
    add_executable(01-heat-conduction-solution playground/01-heat-conduction-solution.cu)
    set_property(TARGET 01-heat-conduction 01-heat-conduction-solution
        PROPERTY FOLDER "Playground/01")

    add_executable(02-hi playground/02-hi.cu)
    set_property(TARGET 02-hi
        PROPERTY FOLDER "Playground/02")

###############################################################################
# Streams
###############################################################################
add_executable(03-stream_legacy playground/03-multistream.cu)
add_executable(03-stream_per_thread playground/03-multistream.cu)
target_compile_options(03-stream_per_thread PRIVATE --default-stream per-thread)
set_property(TARGET 03-stream_legacy 03-stream_per_thread
    PROPERTY FOLDER "Playground/Streams")

add_executable(04-stdthreads_legacy playground/04-thread_test.cu)
add_executable(04-stdthreads_per_thread playground/04-thread_test.cu)
target_compile_options(04-stdthreads_per_thread PRIVATE --default-stream per-thread)

set_property(TARGET 04-stdthreads_legacy 04-stdthreads_per_thread
    PROPERTY FOLDER "Playground/Streams")
endif()

###############################################################################
# Octo-tiger Radiation Kernel
###############################################################################
add_executable(radiation_kernel
    radiation/main.cpp
    radiation/config.hpp
    radiation/config.cpp
    radiation/fx_case.hpp
    radiation/fx_case.cpp
    radiation/kernel_cpu.hpp
    radiation/kernel_cpu.cpp
    radiation/kernel_gpu.hpp
    radiation/kernel_gpu.cu
)
#target_compile_options(radiation_kernel PRIVATE --default-stream per-thread)
target_compile_definitions(radiation_kernel PRIVATE _USE_MATH_DEFINES)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

set_property(TARGET radiation_kernel
    PROPERTY FOLDER "Octo-tiger")

###############################################################################
# Default Project
###############################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY
    VS_STARTUP_PROJECT radiation)
    #VS_STARTUP_PROJECT 01-heat-conduction)

#if(BUILD_TESTING)
#  add_executable(cip_test_1 test_add_integ.cu add_integ.cu)
#
#  add_test(NAME add_integ COMMAND cip_test_1)
#endif()