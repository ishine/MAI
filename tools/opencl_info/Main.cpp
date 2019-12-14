#include "OpenCLHandle.h"
#include <sstream>

std::string vectorToString1(std::vector<size_t> values) {
    std::stringstream ss;
    for (size_t v : values) {
        ss << v << ", ";
    }
    return ss.str();
}

std::string boolToString(cl_bool v) {
    if (v == CL_TRUE) {
        return "TRUE";
    } else if (v == CL_FALSE) {
        return "FALSE";
    }
    return "";
}

std::string deviceTypeToString(cl_device_type type) {
    if (type == CL_DEVICE_TYPE_DEFAULT) {
        return "CL_DEVICE_TYPE_DEFAULT";
    } else if (type == CL_DEVICE_TYPE_CPU) {
        return "CL_DEVICE_TYPE_CPU";
    } else if (type == CL_DEVICE_TYPE_GPU) {
        return "CL_DEVICE_TYPE_GPU";
    } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
        return "CL_DEVICE_TYPE_ACCELERATOR";
    } else if (type == CL_DEVICE_TYPE_CUSTOM) {
        return "CL_DEVICE_TYPE_CUSTOM";
    }
    return "Unknown";
}

std::string memStr(long size) {
    std::stringstream ss;
    ss << size << "B";
    if ((size /= 1024) != 0) {
        ss.str("");
        ss << size << "KB";
    }
    if ((size /= 1024) != 0) {
        ss.str("");
        ss << size << "MB";
    }

    if ((size /= 1024) != 0) {
        ss.str("");
        ss << size << "GB";
    }
    return ss.str();
}

void getInfoByDevice(cl::Device& device, int index) {
    printf("  Device%d {\n", index);
    printf("    name: %s\n", device.getInfo<CL_DEVICE_NAME>().c_str());
    printf("    type: %s\n", deviceTypeToString(device.getInfo<CL_DEVICE_TYPE>()).c_str());
    printf("    vendor: %s\n", device.getInfo<CL_DEVICE_VENDOR>().c_str());
    printf("    version: %s\n", device.getInfo<CL_DEVICE_VERSION>().c_str());
    printf("    extersions: %s\n", device.getInfo<CL_DEVICE_EXTENSIONS>().c_str());
    printf("    opencl_c_version: %s\n", device.getInfo<CL_DEVICE_OPENCL_C_VERSION>().c_str());
    printf("    CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    printf("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %u\n", device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>());
    printf("    CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES: %s\n", vectorToString1(device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()).c_str());
    printf("    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %s\n", memStr(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()).c_str());
    printf("    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %s\n", memStr(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()).c_str());
    printf("    CL_DEVICE_GLOBAL_MEM_SIZE: %s\n", memStr(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()).c_str());
    printf("    CL_DEVICE_LOCAL_MEM_SIZE: %s\n", memStr(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()).c_str());
    printf("    CL_DEVICE_MAX_PARAMETER_SIZE: %lu\n", device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>());
    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: %u\n", device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>());
    printf("    CL_DEVICE_PROFILING_TIMER_RESOLUTION: %uns\n", device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>());
    printf("    CL_DEVICE_IMAGE_SUPPORT: %s\n", boolToString(device.getInfo<CL_DEVICE_IMAGE_SUPPORT>()).c_str());
    printf("    CL_DEVICE_ENDIAN_LITTLE: %s\n", boolToString(device.getInfo<CL_DEVICE_ENDIAN_LITTLE>()).c_str());
    printf("    CL_DEVICE_AVAILABLE: %s\n", boolToString(device.getInfo<CL_DEVICE_AVAILABLE>()).c_str());
    printf("    CL_DEVICE_COMPILER_AVAILABLE: %s\n", boolToString(device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>()).c_str());
    printf("  }// Device%d\n", index);
}

void getInfoByPlatform(cl::Platform& platform, int index) {
    printf("Platform%d {\n", index);
    printf("  name: %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str());
    printf("  version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str());
    printf("  vendor: %s\n", platform.getInfo<CL_PLATFORM_VENDOR>().c_str());
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    int deviceIndex = 0;
    for (cl::Device& device : devices) {
        getInfoByDevice(device, deviceIndex++);
    }
    printf("}// Platform%d\n", index);
}

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        printf("Cannot support opencl\n");
        return 0;
    }
    int index = 0;
    for (cl::Platform& platform : platforms) {
        getInfoByPlatform(platform, index++);
    }


}
