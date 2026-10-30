// Suite skip: the harness's own example of a check that can't run here. The OptiX suites skip every check on a
// computer whose build or driver has no OptiX, and such a suite must pass (exit 0) with its skips counted on the last
// line, while --require-optix, which every gate passes, makes their fixture fail instead. This suite shows the first
// half on every computer: one SKIP, which no option turns into a failure (it isn't OptiX's), and the two budget checks
// that end every suite. A file of its own, since the existing suite files stay as they are.
#include "gpu_test.h"

namespace
{
GPU_TEST(skip, outcome)
{
    ctx.skip("outcome", "the harness's own example: a check that can't run here is skipped, never failed");
}
}
