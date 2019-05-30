add_executable(01-heat-conduction 01-heat-conduction.cu cuda-gpuinfo.h)
add_executable(01-heat-conduction-solution 01-heat-conduction-solution.cu)
set_property(TARGET 01-heat-conduction 01-heat-conduction-solution
    PROPERTY FOLDER "Playground/01 - Hello World")

add_executable(02-hi 02-hi.cu)
set_property(TARGET 02-hi
    PROPERTY FOLDER "Playground/02 - Hello World")

###########################################################################
# Streams
###########################################################################
add_executable(03-stream_legacy 03-multistream.cu)
add_executable(03-stream_per_thread 03-multistream.cu)
target_compile_options(03-stream_per_thread PRIVATE --default-stream per-thread)
set_property(TARGET 03-stream_legacy 03-stream_per_thread
    PROPERTY FOLDER "Playground/04 - Streams")

add_executable(04-stdthreads_legacy 04-thread_test.cu)
add_executable(04-stdthreads_per_thread 04-thread_test.cu)
target_compile_options(04-stdthreads_per_thread PRIVATE --default-stream per-thread)

set_property(TARGET 04-stdthreads_legacy 04-stdthreads_per_thread
    PROPERTY FOLDER "Playground/04 - Streams")

###########################################################################
# Device Exploration
###########################################################################
add_executable(05-device_info 05-device_info.cu)
add_executable(05-bank_conflict_demo 05-bank_conflict_demo.cu)
set_property(TARGET 05-device_info 05-bank_conflict_demo
    PROPERTY FOLDER "Playground/05 - Device Exploration")

###########################################################################
# Matrix Multiplication
###########################################################################
add_executable(06-matmul 06-matmul.cu)
add_executable(06-matmul_shared_mem 06-matmul_shared_mem.cu)
set_property(TARGET 06-matmul 06-matmul_shared_mem
    PROPERTY FOLDER "Playground/06 - Matrix Multiplication")

###########################################################################
# Callbacks
###########################################################################
add_executable(07-callback 07-callback.cu)
set_property(TARGET 07-callback
    PROPERTY FOLDER "Playground/07 - Callbacks")
