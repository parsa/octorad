# Octo-rad
Octo-tiger's Radiation Kernel

## Resources
* [Archived Dump File (2.8MB)- Dropbox](https://www.dropbox.com/s/zo5q4y5ykl9u2n5/arg-dumps.7z?dl=1) - Extract after download (takes 5.7GB)

## CMake Flags
* `OCTORAD_DUMP_DIR` - Path to directory containing `*.args` and `*.outs` dump files
* `OCTORAD_WITH_VC` - Enable Vc model, requires Vc CMake module to be visible
    * Note: I usually use Vc by specifying `Vc_DIR`
* `OCTORAD_WITH_PLAYGROUND` - Enable old toy kernels unrelated to Octo-tiger

