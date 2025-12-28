rd /s /q cmake-build-debug
cmake -H. -Bcmake-build-debug -G"Visual Studio 15 2017 Win64" -DOCTORAD_WITH_SCRATCHPAD=ON -DOCTORAD_WITH_PLAYGROUND=ON -DCMAKE_TOOLCHAIN_FILE="C:/Repos/vcpkg/scripts/buildsystems/vcpkg.cmake" -DHPX_DIR="C:/Repos/hpx/cmake-build-debug/lib/cmake/HPX"
cmake --build cmake-build-debug --target fidelity_v2_test
pause
cmake --open cmake-build-debug