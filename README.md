# Octo-rad
Octo-tiger's Radiation Kernel

## Resources
* [Archived Dump File (2.8MB)- Dropbox](https://www.dropbox.com/s/zo5q4y5ykl9u2n5/arg-dumps.7z?dl=1) - Extract after download (takes 5.7GB)
    * Note: Extract only indices 0-9: `7z x arg-dumps.7z\?dl\=1 -odumps/ '*-?.*' -r`

## CMake Flags
* `OCTORAD_DUMP_DIR` - Path to directory containing `*.args` and `*.outs` dump files
* `OCTORAD_WITH_VC` - Enable Vc model, requires Vc CMake module to be visible (default: OFF)
    * Note: I usually use Vc by specifying `Vc_DIR`
* `OCTORAD_WITH_PLAYGROUND` - Enable old toy kernels unrelated to Octo-tiger (default: OFF)
* `OCTORAD_WITH_SCRATCHPAD` - Enable the scratchpad project(s) (default: OFF)
* `OCTORAD_WITH_CUDA` - Enable GPU kernels and testing codes (default: ON)

## Example Minimal Setup Commands
```bash
wget -O arg-dumps.7z https://www.dropbox.com/s/zo5q4y5ykl9u2n5/arg-dumps.7z?dl=1
7z x arg-dumps.7z -odumps/ '*-78.*' -r
module load cuda/10.1.105
cmake -H. -Bcmake-build-debug -DOCTORAD_DUMP_DIR=dumps
cmake --build cmake-build-debug
./cmake-build-debug/radiation_kernel
```

