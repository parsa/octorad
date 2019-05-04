# dump directory must exists
if(NOT EXISTS ${OCTORAD_DUMP_DIR} OR NOT IS_DIRECTORY ${OCTORAD_DUMP_DIR})
    message(FATAL_ERROR "Unable to access dump directory.")
endif()
message(STATUS "OCTORAD_DUMP_DIR is set to: " ${OCTORAD_DUMP_DIR})

if(OCTORAD_WITH_CUDA)
    ###########################################################################
    # radiation kernel
    ###########################################################################
    add_library(radiation_kernel
        config.hpp
        kernel_gpu.hpp
        kernel_gpu.cu
    )
    # MSVC-specific <math.h> macro
    if(MSVC)
        target_compile_definitions(radiation_kernel PRIVATE
            _USE_MATH_DEFINES)
    endif()
    #target_compile_features(radiation_kernel PUBLIC cxx_std_14)
    target_compile_options(radiation_kernel PRIVATE --default-stream per-thread -Xptxas -v)
    #set(CMAKE_CUDA_FLAGS "--default-stream per-thread -Xptxas -v")

    # allow CUDA code to call device functions implemented in other translation units
    # improve incremental build performance (only recompile and link modified units)
    #set_target_properties(radiation_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

###############################################################################
# test application
###############################################################################
add_executable(radiation_kernel_tester
    main.cpp
    config.hpp
    util.hpp
    util.cpp
    fx_case.hpp
    fx_case.cpp
    kernel_cpu.hpp
    kernel_cpu.cpp
    kernel_gpu.hpp
)
# MSVC-specific <math.h> macro
if(MSVC)
    target_compile_definitions(radiation_kernel_tester PRIVATE
        _USE_MATH_DEFINES)
endif()
# dump folder macro
target_compile_definitions(radiation_kernel_tester PRIVATE
    OCTORAD_DUMP_DIR="${OCTORAD_DUMP_DIR}")
if(OCTORAD_WITH_CUDA)
    # link with the kernel
    target_link_libraries(radiation_kernel_tester PRIVATE radiation_kernel)
    # cuda enabled macro
    target_compile_definitions(radiation_kernel_tester PRIVATE
        OCTORAD_HAVE_CUDA)
endif()
# Vc integration
if(OCTORAD_WITH_VC)
    target_link_libraries(radiation_kernel_tester PRIVATE octorad_vc)
endif()

# IDE settings
set_property(TARGET radiation_kernel radiation_kernel_tester
    PROPERTY FOLDER "Octo-tiger")