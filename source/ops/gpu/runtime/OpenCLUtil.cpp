#include "OpenCLUtil.h"

namespace MAI {

MAI_STATUS run1DKernel(OpenCLRuntime* runtime, cl::Kernel& kernel,
        std::vector<uint32>& gws, std::vector<uint32>& lws, cl::Event& event) {
    //cl::Event event;
    ALOGI("run1DKernel");
    cl_int ret = runtime->commandQueue().enqueueNDRangeKernel(kernel, 0,
            cl::NDRange(gws[0]), cl::NDRange(lws[0]), NULL, &event);
    if (ret != CL_SUCCESS) {
        ALOGE("enqueueNDRangeKernel error:%s", runtime->openCLErrorCodeToString(ret).c_str());
        return MAI_FAILED;
    }
    ALOGI("run1DKernel wait");
    //ALOGI("run1DKernel end cost:%lu us", (end - start));
    return MAI_SUCCESS;
}

MAI_STATUS run2DKernel(OpenCLRuntime* runtime, cl::Kernel& kernel,
        std::vector<uint32>& gws, std::vector<uint32>& lws) {
    cl::Event event;
    runtime->commandQueue().enqueueNDRangeKernel(kernel, 0,
            cl::NDRange(gws[0], gws[1]), cl::NDRange(lws[0], lws[1]), NULL, &event);
    event.wait();
    return MAI_SUCCESS;
}

std::vector<uint32> default1DLocalWorkSize(OpenCLRuntime* runtime,
        const std::vector<uint32>& gws) {
}


} // namespace MAI
