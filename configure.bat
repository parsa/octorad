rd /s /q cmake-build-debug
::cmake -H. -Bcmake-build-debug -G"Visual Studio 15 2017 Win64" -DOCTORAD_DUMP_DIR=C:/Users/Parsa/Desktop/arg-dumps -DOCTORAD_WITH_VC=ON -DVc_DIR="C:/Repos/Vc/cmake-install-debug/lib/cmake/Vc" -DOCTORAD_WITH_SCRATCHPAD=ON
cmake -H. -Bcmake-build-debug -G"Visual Studio 15 2017 Win64" -DOCTORAD_DUMP_DIR=C:/Users/Parsa/Desktop/arg-dumps -DOCTORAD_WITH_SCRATCHPAD=ON -DOCTORAD_DUMP_COUNT=13140
cmake --build cmake-build-debug --target fidelity_test
pause
cmake --open cmake-build-debug