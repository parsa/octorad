# dump directory must exists
if(NOT EXISTS ${OCTORAD_DUMP_DIR} OR NOT IS_DIRECTORY ${OCTORAD_DUMP_DIR})
    message(FATAL_ERROR "Unable to access dump directory.")
endif()
message(STATUS "OCTORAD_DUMP_DIR is set to: " ${OCTORAD_DUMP_DIR})
message(STATUS "OCTORAD_DUMP_COUNT is set to: " ${OCTORAD_DUMP_COUNT})

###############################################################################
# general configuration
###############################################################################
add_library(global_config INTERFACE)
# MSVC-specific <math.h> macro
if(MSVC)
    target_compile_definitions(global_config INTERFACE
        _USE_MATH_DEFINES)
endif()
# Add headers folder to compiler lookup paths
target_include_directories(global_config INTERFACE
    ${PROJECT_SOURCE_DIR}/radiation)
# Number of dumped cases
target_compile_definitions(global_config INTERFACE
    OCTORAD_DUMP_COUNT=${OCTORAD_DUMP_COUNT})
# Vc integration
if(OCTORAD_WITH_VC)
    target_link_libraries(global_config INTERFACE octorad_vc)
endif()

if(OCTORAD_WITH_CUDA)
    ###########################################################################
    # radiation kernel
    ###########################################################################
    add_library(kernel_gpu
        config.hpp
        kernels/helpers.hpp
        kernels/kernel_gpu.hpp
        kernels/kernel_gpu.cu
    )
    target_link_libraries(kernel_gpu PRIVATE global_config)
    #target_compile_features(kernel_gpu PUBLIC cxx_std_14)
    target_compile_options(kernel_gpu PRIVATE --default-stream per-thread -Xptxas -v)
    #set(CMAKE_CUDA_FLAGS "--default-stream per-thread -Xptxas -v")

    # CUDA enabled macro
    target_compile_definitions(kernel_gpu INTERFACE
        OCTORAD_HAVE_CUDA)

    # allow CUDA code to call device functions implemented in other translation units
    # improve incremental build performance (only recompile and link modified units)
    #set_target_properties(kernel_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET kernel_gpu PROPERTY FOLDER "Octo-tiger/Kernels")
endif()
add_library(kernel_cpu
    config.hpp
    kernels/kernel_cpu.hpp
    kernels/kernel_cpu.cpp
)
target_link_libraries(kernel_cpu PRIVATE global_config)

add_library(kernel_v2
    config.hpp
    kernels/kernel_v2.hpp
    kernels/kernel_v2.cpp
)
target_link_libraries(kernel_v2 PRIVATE global_config)
set_property(TARGET kernel_cpu kernel_v2 PROPERTY FOLDER "Octo-tiger/Kernels")

###############################################################################
# utility libraries
###############################################################################
add_library(case_loader
    utils/util.hpp
    utils/fx_case.hpp
    utils/fx_case.cpp
    utils/fx_compare.hpp
    utils/fx_compare.cpp
)
# dump folder macro
target_compile_definitions(case_loader PRIVATE
    OCTORAD_DUMP_DIR="${OCTORAD_DUMP_DIR}")
target_link_libraries(case_loader PUBLIC global_config)
set_property(TARGET case_loader PROPERTY FOLDER "Octo-tiger/Helpers")

###############################################################################
# run case 83 test application
###############################################################################
if(OCTORAD_WITH_CUDA)
    add_executable(run_83_test
        tests/run_83_test.cu
        config.hpp
        utils/util.hpp
        kernels/kernel_gpu.hpp
    )
    target_link_libraries(run_83_test PRIVATE
        kernel_gpu case_loader)
    set_target_properties(run_83_test PROPERTIES FOLDER "Octo-tiger/Tests/Unit")
endif()

###############################################################################
# kernel fidelity check for case 83 test application
###############################################################################
if(OCTORAD_WITH_CUDA)
    add_executable(fidelity_83_test
        tests/fidelity_83_test.cu
        config.hpp
        utils/util.hpp
        kernels/kernel_gpu.hpp
    )
    target_link_libraries(fidelity_83_test PRIVATE
        kernel_gpu case_loader)
    set_target_properties(fidelity_83_test PROPERTIES FOLDER "Octo-tiger/Tests/Unit")
endif()

###############################################################################
# kernel fidelity test application
###############################################################################
add_executable(fidelity_test
    tests/fidelity_test.cpp
    config.hpp
    utils/util.hpp
    kernels/kernel_cpu.hpp
    kernels/kernel_v2.hpp
    kernels/kernel_gpu.hpp
)
if(OCTORAD_WITH_CUDA)
    # link with the kernel
    target_link_libraries(fidelity_test PRIVATE kernel_gpu)
endif()
target_link_libraries(fidelity_test PRIVATE case_loader kernel_cpu kernel_v2)
set_target_properties(fidelity_test PROPERTIES FOLDER "Octo-tiger/Tests/Integration")

###############################################################################
# kernel profile application
###############################################################################
add_executable(profile_test
    tests/profile_test.cpp
    config.hpp
    utils/util.hpp
    utils/scoped_timer.hpp
    kernels/kernel_cpu.hpp
    kernels/kernel_v2.hpp
    kernels/kernel_gpu.hpp
)
if(OCTORAD_WITH_CUDA)
    # link with the kernel
    target_link_libraries(profile_test PRIVATE kernel_gpu)
endif()
target_link_libraries(profile_test PRIVATE case_loader kernel_cpu kernel_v2)
set_target_properties(profile_test PROPERTIES FOLDER "Octo-tiger/Tests/Integration")

###############################################################################
# args array sizes test
###############################################################################
add_executable(arg_array_sizes_test
    tests/arg_array_sizes_test.cpp
    config.hpp
    utils/util.hpp
)
target_link_libraries(arg_array_sizes_test PRIVATE case_loader)
set_target_properties(arg_array_sizes_test PROPERTIES FOLDER "Octo-tiger/Tests/Data")

###############################################################################
# IDE settings
###############################################################################

