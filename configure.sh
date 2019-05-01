#!/usr/bin/env bash

set -euxo pipefail

DUMP_DIR=~/Repositories/octorad/dumps

cmake -H. -Bcmake-build-debug -GNinja -DOCTORAD_DUMP_DIR=$DUMP_DIR
cmake --build cmake-build-debug

read -p "Press [Enter] key to start backup..."
#cmake --open cmake-build-debug

