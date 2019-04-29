rd /s /q cmake-build-debug
cmake -H. -Bcmake-build-debug -G"Visual Studio 15 2017 Win64" -DBUILD_TESTING=ON
pause
cmake --open cmake-build-debug