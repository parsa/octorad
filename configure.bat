rd /s /q cmake-build-debug
cmake -H. -Bcmake-build-debug -G"Visual Studio 15 2017 Win64" -DOCTORAD_DUMP_DIR=C:/Users/Parsa/Desktop/arg-dumps -DOCTORAD_WITH_VC=OFF -DVc_DIR="C:/Repos/Vc/cmake-install-debug/lib/cmake/Vc"
pause
cmake --open cmake-build-debug