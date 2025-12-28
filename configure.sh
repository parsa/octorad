#!/usr/bin/env bash

set -euxo pipefail

if [[ ! -x $(command -v nvcc) ]]; then
    echo 'cannot find nvcc'
    exit 1
fi
if [[ ! -x $(command -v ninja) ]]; then
    echo 'cannot find ninja'
    exit 1
fi

cmake -H. -Bcmake-build-debug -GNinja -DOCTORAD_WITH_HPX=OFF -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build cmake-build-debug

#read -p "Press [Enter] key to start backup..."
#cmake --open cmake-build-debug

