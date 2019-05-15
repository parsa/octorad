#!/usr/bin/env bash

set -euxo pipefail

DUMP_DIR=${DUMP_DIR:-~/Repositories/octorad/dumps}

if [[ ! -d ${DUMP_DIR} ]]; then
    echo 'Unable to access dump directory.'
    exit 1
fi
if [[ ! -x $(command -v nvcc) ]]; then
    echo 'cannot find nvcc'
    exit 1
fi
if [[ ! -x $(command -v ninja) ]]; then
    echo 'cannot find ninja'
    exit 1
fi

cmake -H. -Bcmake-build-debug -GNinja -DOCTORAD_DUMP_DIR=${DUMP_DIR} -DOCTORAD_DUMP_COUNT=OCTORAD_DUMP_COUNT=13140
cmake --build cmake-build-debug

#read -p "Press [Enter] key to start backup..."
#cmake --open cmake-build-debug

