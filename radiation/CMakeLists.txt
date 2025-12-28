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
target_compile_features(global_config INTERFACE cxx_std_20)
target_compile_definitions(global_config INTERFACE
    OCTORAD_DUMP_COUNT=${OCTORAD_DUMP_COUNT}
    OCTORAD_DUMPS_PACK="${OCTORAD_DUMPS_PACK}")
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
    target_compile_options(kernel_gpu PRIVATE --default-stream per-thread --maxrregcount=112 -Xptxas=-v,--warn-on-spills)

    # Kept for reference
    #target_compile_features(kernel_gpu PUBLIC cxx_std_14)

    # Kept for reference
    #set(CMAKE_CUDA_FLAGS "--default-stream per-thread -Xptxas -v")

    # Target architecture
    target_compile_options(kernel_gpu PRIVATE -arch=sm_70)

    # CUDA enabled macro
    target_compile_definitions(kernel_gpu INTERFACE
        OCTORAD_HAVE_CUDA)

    # Kept for reference
    # Allow CUDA code to call device functions implemented in other translation units
    # improve incremental build performance (only recompile and link modified units)
    #set_target_properties(kernel_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    set_property(TARGET kernel_gpu PROPERTY FOLDER "Octo-tiger/Kernels")

    # Relaxed kernel variant for A/B performance testing
    add_library(kernel_gpu_relaxed
        config.hpp
        kernels/helpers.hpp
        kernels/kernel_gpu.hpp
        kernels/kernel_gpu.cu
    )
    target_link_libraries(kernel_gpu_relaxed PRIVATE global_config)
    target_compile_definitions(kernel_gpu_relaxed PRIVATE OCTORAD_RELAXED=1)
    target_compile_options(kernel_gpu_relaxed PRIVATE --default-stream per-thread -Xptxas=-v)
    target_compile_options(kernel_gpu_relaxed PRIVATE -arch=sm_70)
    target_compile_definitions(kernel_gpu_relaxed INTERFACE OCTORAD_HAVE_CUDA)
    set_property(TARGET kernel_gpu_relaxed PROPERTY FOLDER "Octo-tiger/Kernels")
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
    utils/dump_reader.hpp
    utils/dump_reader.cpp
)
set_property(TARGET case_loader PROPERTY FOLDER "Octo-tiger/Helpers")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD_PKG QUIET libzstd)
endif()
if(ZSTD_PKG_FOUND)
    set(ZSTD_INCLUDE_DIR ${ZSTD_PKG_INCLUDE_DIRS})
    set(ZSTD_LIBRARIES ${ZSTD_PKG_LIBRARIES})
else()
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARIES zstd)
endif()
if(NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIBRARIES)
    message(FATAL_ERROR
        "Unable to find libzstd. Install the development package or set ZSTD_INCLUDE_DIR / ZSTD_LIBRARIES.")
endif()
target_include_directories(case_loader PRIVATE ${ZSTD_INCLUDE_DIR})
target_link_libraries(case_loader PUBLIC global_config PRIVATE ${ZSTD_LIBRARIES})

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
# kernel fidelity test applications
###############################################################################
if(OCTORAD_WITH_CUDA)
    add_executable(fidelity_gpu_test
        tests/fidelity_test.cpp
        config.hpp
        utils/util.hpp
        kernels/kernel_cpu.hpp
        kernels/kernel_gpu.hpp
    )
    target_link_libraries(fidelity_gpu_test PRIVATE
        case_loader kernel_cpu kernel_gpu)
    set_target_properties(fidelity_gpu_test
        PROPERTIES FOLDER "Octo-tiger/Tests/Integration")
endif()

add_executable(fidelity_v2_test
    tests/fidelity_v2_test.cpp
    config.hpp
    utils/util.hpp
    kernels/kernel_cpu.hpp
    kernels/kernel_v2.hpp
)
target_link_libraries(fidelity_v2_test PRIVATE
    case_loader kernel_cpu kernel_v2)
set_target_properties(fidelity_v2_test
    PROPERTIES FOLDER "Octo-tiger/Tests/Integration")

###############################################################################
# compare kernel profile application
###############################################################################
add_executable(compare_test
    tests/compare_test.cpp
    config.hpp
    utils/util.hpp
    utils/scoped_timer.hpp
    kernels/kernel_cpu.hpp
    kernels/kernel_v2.hpp
    kernels/kernel_gpu.hpp
)
if(OCTORAD_WITH_CUDA)
    target_link_libraries(compare_test PRIVATE case_loader kernel_gpu)
endif()
target_link_libraries(compare_test PRIVATE case_loader kernel_cpu kernel_v2)
set_target_properties(compare_test PROPERTIES FOLDER "Octo-tiger/Tests/Profiling")

###############################################################################
# kernel profile (many cases) applications
###############################################################################
set(_profile_many_sources
    tests/profile_many_test.cpp
    config.hpp
    utils/util.hpp
    utils/scoped_timer.hpp
    kernels/kernel_cpu.hpp
    kernels/kernel_v2.hpp
)

add_executable(profile_ref_test ${_profile_many_sources})
target_compile_definitions(profile_ref_test PRIVATE
    PROFILE_VARIANT=PROFILE_VARIANT_CPU_REF)
target_link_libraries(profile_ref_test PRIVATE case_loader kernel_cpu)
set_target_properties(profile_ref_test PROPERTIES FOLDER "Octo-tiger/Tests/Profiling")

add_executable(profile_v2_test ${_profile_many_sources})
target_compile_definitions(profile_v2_test PRIVATE
    PROFILE_VARIANT=PROFILE_VARIANT_V2)
target_link_libraries(profile_v2_test PRIVATE case_loader kernel_v2)
set_target_properties(profile_v2_test PROPERTIES FOLDER "Octo-tiger/Tests/Profiling")

if(OCTORAD_WITH_CUDA)
    add_executable(profile_gpu_test
        ${_profile_many_sources}
        kernels/kernel_gpu.hpp
    )
    target_compile_definitions(profile_gpu_test PRIVATE
        PROFILE_VARIANT=PROFILE_VARIANT_GPU)
    target_link_libraries(profile_gpu_test PRIVATE case_loader kernel_gpu)
    set_target_properties(profile_gpu_test PROPERTIES FOLDER "Octo-tiger/Tests/Profiling")

    add_executable(profile_gpu_relaxed_test
        ${_profile_many_sources}
        kernels/kernel_gpu.hpp
    )
    target_compile_definitions(profile_gpu_relaxed_test PRIVATE
        PROFILE_VARIANT=PROFILE_VARIANT_GPU)
    target_link_libraries(profile_gpu_relaxed_test PRIVATE case_loader kernel_gpu_relaxed)
    set_target_properties(profile_gpu_relaxed_test PROPERTIES FOLDER "Octo-tiger/Tests/Profiling")
endif()

###############################################################################
# kernel profile case 83 application
###############################################################################
if(OCTORAD_WITH_CUDA)
    add_executable(profile_83_test
        tests/profile_83_test.cpp
        config.hpp
        utils/util.hpp
        utils/scoped_timer.hpp
        kernels/kernel_cpu.hpp
        kernels/kernel_v2.hpp
        kernels/kernel_gpu.hpp
    )
    target_link_libraries(profile_83_test PRIVATE case_loader kernel_gpu)
    set_target_properties(profile_83_test PROPERTIES FOLDER "Octo-tiger/Tests/Profiling")
endif()

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

# Make sure executables link against the GCC 14 runtime that matches the chosen toolchain
set(_gcc14_lib64 "/opt/apps/gcc/14.2.0/lib64")
set(_gcc14_libgcc "/opt/apps/gcc/14.2.0/lib/gcc/x86_64-redhat-linux/14.2.0")
set(_stdcxx_targets
    fidelity_gpu_test
    fidelity_v2_test
    compare_test
    profile_ref_test
    profile_v2_test
    arg_array_sizes_test
)
if(OCTORAD_WITH_CUDA)
    list(APPEND _stdcxx_targets
        profile_gpu_test
        profile_gpu_relaxed_test
        profile_83_test
        fidelity_83_test)
endif()
foreach(t IN LISTS _stdcxx_targets)
    if(TARGET ${t})
        target_link_directories(${t} PRIVATE "${_gcc14_lib64}" "${_gcc14_libgcc}")
        target_link_options(${t} PRIVATE
            -L${_gcc14_lib64}
            -Wl,-rpath,${_gcc14_lib64}
            -Wl,-rpath,${_gcc14_libgcc})
        target_link_libraries(${t} PRIVATE /opt/apps/gcc/14.2.0/lib64/libstdc++.so)
    endif()
endforeach()

###############################################################################
# IDE settings
###############################################################################
