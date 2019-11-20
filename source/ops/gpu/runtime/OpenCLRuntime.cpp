#include "OpenCLRuntime.h"
#include "OpenCLHandle.h"
#include "util/MAIType.h"
#include "util/MAIUtil.h"
#include <iostream>
#include <fstream>
#include <memory>

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

MAI_STATUS CLPrecompiledFile::load() {
    std::ifstream f(mFile.c_str());
    if (!f.good()) {
        return MAI_FAILED;
    }

    const char* data = mapFile<char>(mFile);
    uint64 dataNum = 0;
    memcpy(&dataNum, data, sizeof(uint64));
    data += sizeof(uint64);
    int32 intSize = sizeof(int32);
    int32 keySize = 0;
    int32 valueSize = 0;
    for (uint64 i = 0; i < dataNum; ++i) {
        memcpy(&keySize, data, intSize);
        data += intSize;

        std::unique_ptr<char[]> key(new char[keySize+1]);
        memcpy(&key[0], data, keySize);
        key[keySize] = '\0';
        data += keySize;

        memcpy(&valueSize, data, intSize);
        data += intSize;

        std::vector<unsigned char> value(valueSize);
        memcpy(value.data(), data, valueSize);
        data += valueSize;

        mData.emplace(std::string(&key[0]), value);
    }
    return MAI_SUCCESS;
}

MAI_STATUS CLPrecompiledFile::store() {
    std::ofstream outFile(mFile, std::ios::out | std::ios::binary);
    uint64 dataSize = sizeof(uint64);
    for (auto& kv : mData) {
        dataSize += sizeof(uint32) * 2 + kv.first.size() + kv.second.size();
    }
    std::unique_ptr<char[]> buffer(new char[dataSize]);
    char* bufferPtr = buffer.get();
    uint64 size = mData.size();
    memcpy(bufferPtr, &size, sizeof(uint64));
    bufferPtr += sizeof(uint64);
    for (auto& kv : mData) {
        int32 keySize = kv.first.size();
        memcpy(bufferPtr, &keySize, sizeof(int32));
        bufferPtr += sizeof(int32);

        memcpy(bufferPtr, kv.first.c_str(), kv.first.size());
        bufferPtr += kv.first.size();

        int32 valueSize = kv.second.size();
        memcpy(bufferPtr, &valueSize, sizeof(int32));
        bufferPtr += sizeof(int32);

        memcpy(bufferPtr, kv.second.data(), kv.second.size());
        bufferPtr += kv.second.size();
    }
    outFile.write(buffer.get(), dataSize);
    return MAI_SUCCESS;
}

void CLPrecompiledFile::add(const std::string& key,
        const std::vector<uint8>& value) {
    auto res = mData.emplace(key, value);
    if (!res.second) {
        mData[key] = value;
    }
}

std::vector<uint8>* CLPrecompiledFile::find(const std::string& key) {
    auto it = mData.find(key);
    if (it == mData.end()) {
        return NULL;
    }

    return &(it->second);
}

OpenCLRuntime::OpenCLRuntime()
    : mDeviceType(OpenCLDeviceType::UNKNOWN),
      mVersionType(OpenCLVersionType::UNKNOWN),
      mCLDir("/sdcard"),
      mPrecompiledFile(mCLDir + "/mai_precompiled_cl.bin") {
    //CLPrecompiledFile file(mCLDir + "/mai_precompiled_cl.bin");
    mPrecompiledFile.load();
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
    std::vector<cl_context_properties> contextProperties(5);
    contextProperties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
    contextProperties.push_back(CL_PERF_HINT_HIGH_QCOM);
    contextProperties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
    contextProperties.push_back(CL_PRIORITY_HINT_HIGH_QCOM);
    contextProperties.push_back(0);
    cl_int err;
    ALOGI("before create context");
    mContext.reset(new cl::Context(devices, contextProperties.data(), NULL, NULL, &err));
    ALOGI("after create context");
    if (err != CL_SUCCESS) {
        MAI_ABORT("Cannot create gpu context: %s", openCLErrorCodeToString(err).c_str());
        return;
    }
    cl_command_queue_properties properties = 0;
    properties |= CL_QUEUE_PROFILING_ENABLE;
    mCommandQueue = std::make_shared<cl::CommandQueue>(*mContext, *mDevice,
            properties/*properties*/, &err/*error*/);

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
    mPrecompiledFile.store();
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

void OpenCLRuntime::setCLDir(const std::string& dir) {
    mCLDir = dir;
}

std::string OpenCLRuntime::getCLDir() {
    return mCLDir;
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
    //if (it != mProgramMap.end()) {
    //    program = it->second;
    //} else {
        bool success = buildProgram(programName, buildOptions, program);
        if (!success) {
            ALOGE("build program(%s, %s) failed", programName.c_str(),
                    buildOptions.c_str());
            return false;
        }
        mProgramMap.emplace(programKey, program);
    //}

    cl_int err;
    kernel = cl::Kernel(program, kernelName.c_str(), &err);
    MAI_CHECK(err == 0, "create kernel %s failed", kernelName.c_str());
    return true;
}

bool OpenCLRuntime::buildProgram(const std::string& programName,
        const std::string& buildOptions,
        cl::Program& program) {
    std::string programUniqueKey = programName + buildOptions;
    bool ret;
    ret = buildProgramFromBinary(programUniqueKey, buildOptions, program);
    if (ret) {
        return true;
    }
    ret = buildProgramFromSource(programUniqueKey, programName, buildOptions, program);
    if (ret) {
        return true;
    }
    return false;
}

bool OpenCLRuntime::buildProgramFromSource(
        const std::string& programKey,
        const std::string& programName,
        const std::string& buildOptions,
        cl::Program& program) {
    std::ifstream programFile(mCLDir + "/" + programName + ".cl");
    std::string programString(std::istreambuf_iterator<char>(programFile),
            (std::istreambuf_iterator<char>()));
    program =  cl::Program(*mContext, programString);
    cl_int ret = program.build({device()}, buildOptions.c_str());
    if (ret != CL_SUCCESS) {
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) == CL_BUILD_ERROR) {
            std::string errLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
            ALOGE("build program(%s) log:", programName.c_str(), errLog.c_str());
        }
        ALOGE("build program(%s) error:%d", programName.c_str(), ret);
        MAI_ABORT("Cannot build program");
        return false;
    }
    std::vector<std::vector<unsigned char>> binary = program.getInfo<CL_PROGRAM_BINARIES>();
    MAI_CHECK(binary.size() == 1, "getInfo<CL_PROGRAM_BINARIES> error");
    mPrecompiledFile.add(programKey, binary[0]);
    ALOGD("build program(%s) from source success", programName.c_str());
    //write program into file
    return true;;
}

bool OpenCLRuntime::buildProgramFromBinary(const std::string& programKey,
        const std::string& buildOptions,
        cl::Program& program) {
    auto content = mPrecompiledFile.find(programKey);
    if (content == NULL) {
        return false;
    }

    cl_int ret;
    program = cl::Program(*mContext, {*mDevice}, {*content});
    ret = program.build({device()}, buildOptions.c_str());
    if (ret != CL_SUCCESS) {
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
                CL_BUILD_ERROR) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
            ALOGI("Program build log:%s", buildLog.c_str());
        }
        ALOGI("Build program from binary error:%s", programKey.c_str());
        return false;
    }
    ALOGD("build program(%s) from binary success", programKey.c_str());
    return true;
}

/*static*/
std::string OpenCLRuntime::openCLErrorCodeToString(cl_int err) {
  switch (err) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
#endif
    default:
      return "UNKNOWN: " + err;
  }
}


}
