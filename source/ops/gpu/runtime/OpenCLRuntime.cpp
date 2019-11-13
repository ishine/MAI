#include "OpenCLRuntime.h"
#include "OpenCLHandle.h"
#include "util/MAIType.h"

namespace {

MAI::OpenCLRuntime::OpenCLDeviceType getDeviceType(const std::string& deviceName) {
    if (deviceName == "QUALCOMM Adreno(TM)") {
        return MAI::OpenCLRuntime::OpenCLDeviceType::QCOM;
    } else if (deviceName == "Mali") {
        return MAI::OpenCLRuntime::OpenCLDeviceType::MALI;
    } else if (deviceName == "PowerVR") {
        return MAI::OpenCLRuntime::OpenCLDeviceType::POWER_VR;
    } else {
        return MAI::OpenCLRuntime::OpenCLDeviceType::UNKNOWN;
    }
}

MAI::OpenCLRuntime::OpenCLVersionType getVersionType(const std::string& versionName) {
    if (versionName == "2.0") {
        return MAI::OpenCLRuntime::OpenCLVersionType::VER_2_2;
    } else if (versionName == "2.0") {
        return MAI::OpenCLRuntime::OpenCLVersionType::VER_2_0;
    } else if (versionName == "1.2") {
        return MAI::OpenCLRuntime::OpenCLVersionType::VER_1_2;
    } else {
        return MAI::OpenCLRuntime::OpenCLVersionType::UNKNOWN;
    }
}

} // namespace

namespace MAI {

OpenCLRuntime::OpenCLRuntime()
    : mDeviceType(OpenCLDeviceType::UNKNOWN),
      mVersionType(OpenCLVersionType::UNKNOWN) {
    // 1. get all platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    MAI_CHECK(!platforms.empty(), "Cannot find opencl platform");
    ALOGI("OpenCLRuntime::OpenCLRuntime platforms size:%d", platforms.size());
    // choose the first platform as only one platform normally
    cl::Platform platform = platforms[0];

    ALOGI("Choose platform name:%s", platform.getInfo<CL_PLATFORM_NAME>().c_str());
    ALOGI("Choose platform version:%s", platform.getInfo<CL_PLATFORM_VERSION>().c_str());

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    MAI_CHECK(!devices.empty(), "Cannot find opencl devices");

    mDevice = std::make_shared<cl::Device>(devices[0]);
    mContext = std::make_shared<cl::Context>(devices);
    cl_int err;
    mCommandQueue = std::make_shared<cl::CommandQueue>(*mContext, *mDevice,
            0/*propertes*/, &err/*error*/);

    std::vector<cl::Device> ctxDevices = mContext->getInfo<CL_CONTEXT_DEVICES>();
    for (int i = 0; i < ctxDevices.size(); ++i) {
        ALOGD("deviceName:%s", ctxDevices[i].getInfo<CL_DEVICE_NAME>().c_str());
        ALOGD("versionName:%s", ctxDevices[i].getInfo<CL_DEVICE_VERSION>().c_str());
    }
    mDeviceType = getDeviceType(ctxDevices[0].getInfo<CL_DEVICE_NAME>());
    mVersionType = getVersionType(ctxDevices[0].getInfo<CL_DEVICE_VERSION>());
}

OpenCLRuntime::~OpenCLRuntime() {
    mCommandQueue->finish();
}

OpenCLRuntime::OpenCLDeviceType OpenCLRuntime::deviceType() {
    return mDeviceType;
}

OpenCLRuntime::OpenCLVersionType OpenCLRuntime::versionType() {
    return mVersionType;
}

cl::Context& OpenCLRuntime::context() {
    return *mContext;
}

cl::Device& OpenCLRuntime::device() {
    return *mDevice;
}

cl::CommandQueue& OpenCLRuntime::commandQueue() {
    return *mCommandQueue;
}

//TODO:(gavinchen) Thread-Safe
bool OpenCLRuntime::buildKernel(
        const std::string& programName,
        const std::string& kernelName,
        const std::string& buildOptions,
        cl::Kernel& kernel) {
    const std::string programKey = programName + buildOptions;
    auto it = mProgramMap.find(programKey);
    cl::Program program;
    if (it != mProgramMap.end()) {
        program = it->second;
    } else {
        bool success = buildProgram(programName, buildOptions, program);
        if (!success) {
            ALOGE("build program(%s, %s) failed", programName.c_str(),
                    buildOptions.c_str());
            return false;
        }
        mProgramMap.emplace(programKey, program);
    }

    cl_int err;
    kernel = cl::Kernel(program, kernelName.c_str(), &err);
    MAI_CHECK(err == 0, "create kernel %s failed", kernelName.c_str());
    return true;
}

bool OpenCLRuntime::buildProgram(const std::string& programName,
        const std::string& buildOptions,
        cl::Program& program) {
    return false;
}

bool OpenCLRuntime::buildProgramFromSource(const std::string& programName,
        const std::string& buildOptions,
        cl::Program& program) {
    return false;
}

bool OpenCLRuntime::buildProgramFromBinary(const std::string& programName,
        const std::string& buildOptions,
        cl::Program& program) {
    return false;
}

}
