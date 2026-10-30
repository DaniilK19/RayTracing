// GpuTests in a build without OptiX's headers: an empty check module in place of optix_check_programs.cu's, so the
// OptiX suites link as they are and SKIP, saying why (start_optix() throws "this build has no OptiX ...").
#include <cstddef>

#include "optix/embedded.h"

extern "C" const unsigned char optix_check_ptx[] = {0}; // no PTX: a C++ array has at least one element
extern "C" const std::size_t optix_check_ptx_size = 0;
extern "C" const char optix_check_ptx_sha256[] = "";
