# dump directory must exists
if(NOT EXISTS ${OCTORAD_DUMP_DIR} OR NOT IS_DIRECTORY ${OCTORAD_DUMP_DIR})
    message(FATAL_ERROR "Unable to access dump directory.")
endif()
message(STATUS "OCTORAD_DUMP_DIR is set to: " ${OCTORAD_DUMP_DIR})

###############################################################################
# general configuration
###############################################################################
add_library(global_config INTERFACE)
# MSVC-specific <math.h> macro
if(MSVC)
    target_compile_definitions(global_config INTERFACE
        _USE_MATH_DEFINES)
endif()
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
        kernel_gpu.hpp
        kernel_gpu.cu
    )
    target_link_libraries(kernel_gpu PRIVATE global_config)
    #target_compile_features(kernel_gpu PUBLIC cxx_std_14)
    target_compile_options(kernel_gpu PRIVATE --default-stream per-thread -Xptxas -v)
    #set(CMAKE_CUDA_FLAGS "--default-stream per-thread -Xptxas -v")

    # allow CUDA code to call device functions implemented in other translation units
    # improve incremental build performance (only recompile and link modified units)
    #set_target_properties(kernel_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET kernel_gpu PROPERTY FOLDER "Octo-tiger/Kernels")
endif()
add_library(kernel_cpu
    config.hpp
    kernel_cpu.hpp
    kernel_cpu.cpp
)
target_link_libraries(kernel_cpu PRIVATE global_config)

add_library(kernel_v2
    config.hpp
    kernel_v2.hpp
    kernel_v2.cpp
)
target_link_libraries(kernel_v2 PRIVATE global_config)

set_property(TARGET kernel_cpu kernel_v2 PROPERTY FOLDER "Octo-tiger/Kernels")

###############################################################################
# utility libraries
###############################################################################
add_library(case_loader
    util.hpp
    fx_case.hpp
    fx_case.cpp
    fx_compare.hpp
    fx_compare.cpp
)
# dump folder macro
target_compile_definitions(case_loader PRIVATE
    OCTORAD_DUMP_DIR="${OCTORAD_DUMP_DIR}")

###############################################################################
# kernel fidelity test application
###############################################################################
add_executable(fidelity_test
    fidelity_test.cpp
    config.hpp
    util.hpp
    kernel_cpu.hpp
    kernel_v2.hpp
    kernel_gpu.hpp
)
if(OCTORAD_WITH_CUDA)
    # link with the kernel
    target_link_libraries(fidelity_test PRIVATE kernel_gpu)
    # cuda enabled macro
    target_compile_definitions(fidelity_test PRIVATE
        OCTORAD_HAVE_CUDA)
endif()
target_link_libraries(fidelity_test PRIVATE case_loader kernel_cpu kernel_v2)

###############################################################################
# kernel profile application
###############################################################################
add_executable(profile_test
    profile_test.cpp
    config.hpp
    util.hpp
    scoped_timer.hpp
    kernel_cpu.hpp
    kernel_v2.hpp
    kernel_gpu.hpp
)
if(OCTORAD_WITH_CUDA)
    # link with the kernel
    target_link_libraries(profile_test PRIVATE kernel_gpu)
    # cuda enabled macro
    target_compile_definitions(profile_test PRIVATE
        OCTORAD_HAVE_CUDA)
endif()
target_link_libraries(profile_test PRIVATE case_loader kernel_cpu kernel_v2)

###############################################################################
# IDE settings
###############################################################################
set_property(TARGET fidelity_test profile_test case_loader PROPERTY FOLDER "Octo-tiger")

