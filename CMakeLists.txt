cmake_minimum_required(VERSION 3.10)
project(octorad CXX)

###############################################################################
# Options
###############################################################################
option(OCTORAD_WITH_PLAYGROUND "Enable old toy kernels unrelated to Octo-tiger (default: OFF)" OFF)
option(OCTORAD_WITH_SCRATCHPAD "Enable the scratchpad project(s) (default: OFF)" OFF)
option(OCTORAD_WITH_VC "Enable Vc model, requires Vc CMake module to be visible (default: OFF)" OFF)
option(OCTORAD_WITH_CUDA "Enable GPU kernels and testing codes (default: ON)" ON)
option(OCTORAD_WITH_HPX "Enable GPU kernels and testing codes (default: ON)" ON)

set(OCTORAD_DUMP_COUNT "" CACHE STRING "Number of dumped cases")
set(OCTORAD_DUMP_DIR "" CACHE PATH "Path to directory containing `*.args` and `*.outs` dump files")

###############################################################################
# CUDA Project
###############################################################################
if(OCTORAD_WITH_CUDA)
    enable_language(CUDA)
else()
    message(STATUS "Octo-rad is being configured without CUDA")
endif()

###############################################################################
# Enable C++14
###############################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# Output Directories
###############################################################################
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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
# HPX
###############################################################################
if(OCTORAD_WITH_HPX)
    find_package(HPX REQUIRED)
endif()

###############################################################################
# Testpad
###############################################################################
if(OCTORAD_WITH_CUDA AND OCTORAD_WITH_SCRATCHPAD)
    add_executable(scratchpad-01 scratchpad/scratchpad-01.cu)
    add_executable(scratchpad-02 scratchpad/scratchpad-02.cu)
    add_executable(scratchpad-03 scratchpad/scratchpad-03.cu)
    add_executable(scratchpad-04 scratchpad/scratchpad-04.cu)
    set_property(TARGET scratchpad-01 scratchpad-02 scratchpad-03 scratchpad-04
        PROPERTY FOLDER "Scratchpad")
endif()

###############################################################################
# Playground
###############################################################################
if(OCTORAD_WITH_CUDA AND OCTORAD_WITH_PLAYGROUND)
    add_subdirectory(playground)
endif()

###############################################################################
# Octo-tiger Radiation Kernel
###############################################################################
add_subdirectory(radiation)

###############################################################################
# Default Project
###############################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY
    VS_STARTUP_PROJECT fidelity_test)

###############################################################################
# Tests
###############################################################################
#if(BUILD_TESTING)
#  add_executable(cip_test_1 test_add_integ.cu add_integ.cu)
#
#  add_test(NAME add_integ COMMAND cip_test_1)
#endif()