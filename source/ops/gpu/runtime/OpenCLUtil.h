#include "OpenCLRuntime.h"

namespace MAI {

MAI_STATUS run1DKernel(OpenCLRuntime* runtime, cl::Kernel& kernel,
        std::vector<uint32>& gws, std::vector<uint32>& lws, cl::Event& event);

MAI_STATUS run2DKernel(OpenCLRuntime* runtime, cl::Kernel& kernel,
        std::vector<uint32>& gws, std::vector<uint32>& lws);

std::vector<uint32> default1DLocalWorkSize(OpenCLRuntime* runtime,
        const std::vector<uint32>& gws);

} // namespace MAI
