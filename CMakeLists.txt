cmake_minimum_required(VERSION 3.10)
project(octorad CXX CUDA)

###############################################################################
# Options
###############################################################################
option(OCTORAD_WITH_PLAYGROUND "" OFF)
option(OCTORAD_WITH_VC "" OFF)

###############################################################################
# Enable C++14
###############################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# Vc
###############################################################################
if(OCTORAD_WITH_VC)
    find_package(Vc REQUIRED)

    add_library(octorad_vc INTERFACE)
    target_compile_definitions(octorad_vc INTERFACE OCTORAD_HAVE_VC)
    target_link_libraries(octorad_vc INTERFACE Vc::Vc)
    # NOTE: No clue how to pass host compiler flags to nvcc in CMake
    if(USE_AVX2)
        if(MSVC)
            target_compile_options(octorad_vc INTERFACE -Xcompiler /arch:AVX2)
        else()
            target_compile_options(octorad_vc INTERFACE -Xcompiler -mavx2)
        endif()
        message(STATUS "Enabled AVX2")
    endif()
endif()

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

if(NOT EXISTS ${OCTORAD_DUMP_DIR} OR NOT IS_DIRECTORY ${OCTORAD_DUMP_DIR})
    message(FATAL_ERROR "Unable to access dump directory.")
endif()
message(STATUS "OCTORAD_DUMP_DIR is set to: " ${OCTORAD_DUMP_DIR})

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
#target_compile_features(particles PUBLIC cxx_std_17)
#target_compile_options(radiation_kernel PRIVATE --default-stream per-thread)
target_compile_definitions(radiation_kernel PRIVATE
    _USE_MATH_DEFINES OCTORAD_DUMP_DIR="${OCTORAD_DUMP_DIR}")
if(OCTORAD_WITH_VC)
    target_link_libraries(radiation_kernel PRIVATE octorad_vc)
endif()
target_compile_options(radiation_kernel INTERFACE --default-stream per-thread)
# improve incremental build performance (only recompile and link modified units)
#set_target_properties(radiation_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET radiation_kernel
    PROPERTY FOLDER "Octo-tiger")

###############################################################################
# Default Project
###############################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY
    VS_STARTUP_PROJECT radiation_kernel)

###############################################################################
# Tests
###############################################################################
#if(BUILD_TESTING)
#  add_executable(cip_test_1 test_add_integ.cu add_integ.cu)
#
#  add_test(NAME add_integ COMMAND cip_test_1)
#endif()