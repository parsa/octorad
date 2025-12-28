# Octo-rad
Octo-tiger's Radiation Kernel

## Resources
* [Indexed Dump Pack (~550MB)](https://github.com/parsa/octorad/releases/download/args-dumps/arg-dumps.pack.zst) - CMake downloads this automatically into the build directory; all tests stream cases directly from this pack

## CMake Flags
* `OCTORAD_WITH_PLAYGROUND` - Enable old toy kernels unrelated to Octo-tiger (default: OFF)
* `OCTORAD_WITH_SCRATCHPAD` - Enable the scratchpad project(s) (default: OFF)
* `OCTORAD_WITH_CUDA` - Enable GPU kernels and testing codes (default: ON)
* `OCTORAD_DUMPS_PACK` - Override path to `arg-dumps.pack.zst`; defaults to `arg-dumps.pack.zst` inside the build tree. If the pack is absent, Octo-rad will build it locally the first time you run any test.
* `OCTORAD_PROFILE_CASES` (env var) - Limit how many cases each `profile_*_test` processes; defaults to all available cases.

## Test Targets
* `fidelity_v2_test`: compares the CPU v2 kernel against the reference CPU kernel (1% of the cases by default).
* `fidelity_gpu_test`: compares the CUDA kernel against the CPU reference (requires building with CUDA enabled).
* `profile_ref_test`, `profile_v2_test`, `profile_gpu_test`: throughput benchmarks for the reference CPU, v2 CPU, and CUDA kernels respectively. Set `OCTORAD_PROFILE_CASES` to shorten runs when needed.

## Example Minimal Setup Commands
```bash
module purge
module load cuda/12.8
module load openmpi/5.0.7
module load cmake/3.29.2
module load gcc/14.2.0
module load ninja/1.13.1
module load zstd # or ensure libzstd headers/libraries are available
cmake -H. -Bcmake-build-debug
cmake --build cmake-build-debug --target fidelity_v2_test
./cmake-build-debug/fidelity_v2_test
# GPU builds can also run ./cmake-build-debug/fidelity_gpu_test and profile_gpu_test
```
