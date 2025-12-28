cmake_minimum_required(VERSION 3.10)
project(octorad CXX)

###############################################################################
# Options
###############################################################################
option(OCTORAD_WITH_PLAYGROUND "Enable old toy kernels unrelated to Octo-tiger (default: OFF)" OFF)
option(OCTORAD_WITH_SCRATCHPAD "Enable the scratchpad project(s) (default: OFF)" OFF)
option(OCTORAD_WITH_CUDA "Enable GPU kernels and testing codes (default: ON)" ON)
option(OCTORAD_WITH_HPX "Enable GPU kernels and testing codes (default: ON)" ON)

set(OCTORAD_DUMPS_PACK "" CACHE PATH "Path to dump pack (.pack.zst)")
set(OCTORAD_DUMPS_PACK_URL
    "https://github.com/parsa/octorad/releases/download/args-dumps/arg-dumps.pack.zst"
    CACHE STRING "URL used to download arg-dumps.pack.zst when no pack path is provided")

if("${OCTORAD_DUMPS_PACK}" STREQUAL "")
    set(OCTORAD_DUMPS_PACK "${CMAKE_BINARY_DIR}/arg-dumps.pack.zst"
        CACHE PATH "Path to dump pack (.pack.zst)" FORCE)
endif()

if(OCTORAD_DUMPS_PACK_URL AND NOT EXISTS "${OCTORAD_DUMPS_PACK}")
    get_filename_component(_octorad_pack_dir "${OCTORAD_DUMPS_PACK}" DIRECTORY)
    if(_octorad_pack_dir)
        file(MAKE_DIRECTORY "${_octorad_pack_dir}")
    endif()
    message(STATUS "Downloading dump pack from ${OCTORAD_DUMPS_PACK_URL}")
    file(DOWNLOAD "${OCTORAD_DUMPS_PACK_URL}" "${OCTORAD_DUMPS_PACK}"
        SHOW_PROGRESS
        STATUS _octorad_pack_status
        LOG _octorad_pack_log)
    list(GET _octorad_pack_status 0 _octorad_pack_status_code)
    if(NOT _octorad_pack_status_code EQUAL 0)
        message(WARNING "Failed to download dump pack: ${_octorad_pack_log}")
    endif()
endif()

# Derive the dump count from the pack header so users never have to touch it.
if(NOT EXISTS "${OCTORAD_DUMPS_PACK}")
    message(FATAL_ERROR
        "Dump pack \"${OCTORAD_DUMPS_PACK}\" is missing. Delete it and rerun CMake to re-download.")
endif()
file(READ "${OCTORAD_DUMPS_PACK}" _octorad_pack_header_hex LIMIT 12 HEX)
string(LENGTH "${_octorad_pack_header_hex}" _octorad_pack_header_len)
if(_octorad_pack_header_len LESS 24)
    message(FATAL_ERROR
        "Dump pack \"${OCTORAD_DUMPS_PACK}\" is truncated (got ${_octorad_pack_header_len} hex chars). Re-download it.")
endif()
set(_octorad_pack_header_bytes "")
foreach(__oct_i RANGE 0 11)
    math(EXPR __oct_start "${__oct_i} * 2")
    string(SUBSTRING "${_octorad_pack_header_hex}" ${__oct_start} 2 __oct_byte)
    list(APPEND _octorad_pack_header_bytes "${__oct_byte}")
endforeach()
function(_octorad_u32_from_bytes result start_index)
    math(EXPR _idx0 "${start_index}")
    math(EXPR _idx1 "${start_index} + 1")
    math(EXPR _idx2 "${start_index} + 2")
    math(EXPR _idx3 "${start_index} + 3")
    list(GET _octorad_pack_header_bytes ${_idx0} b0)
    list(GET _octorad_pack_header_bytes ${_idx1} b1)
    list(GET _octorad_pack_header_bytes ${_idx2} b2)
    list(GET _octorad_pack_header_bytes ${_idx3} b3)
    math(EXPR value "0x${b0} + (0x${b1} << 8) + (0x${b2} << 16) + (0x${b3} << 24)")
    set(${result} "${value}" PARENT_SCOPE)
endfunction()
_octorad_u32_from_bytes(_octorad_pack_magic 0)
if(NOT _octorad_pack_magic EQUAL 0x4f44504b)
    message(FATAL_ERROR
        "Dump pack \"${OCTORAD_DUMPS_PACK}\" has unexpected magic 0x${_octorad_pack_magic}.")
endif()
_octorad_u32_from_bytes(_octorad_pack_version 4)
if(NOT _octorad_pack_version EQUAL 2)
    message(FATAL_ERROR
        "Dump pack \"${OCTORAD_DUMPS_PACK}\" has unsupported version ${_octorad_pack_version}.")
endif()
_octorad_u32_from_bytes(_octorad_pack_entry_count 8)
set(_octorad_pack_header_bytes) # cleanup temporary list

set(OCTORAD_DUMP_COUNT "${_octorad_pack_entry_count}")

###############################################################################
# CUDA Project
###############################################################################
if(OCTORAD_WITH_CUDA)
    enable_language(CUDA)
else()
    message(STATUS "Octo-rad is being configured without CUDA")
endif()

###############################################################################
# Enable C++20
###############################################################################
set(CMAKE_CXX_STANDARD 20)
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