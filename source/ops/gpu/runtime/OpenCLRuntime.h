#include <map>
#include "CL/cl2.hpp"
#include "include/Runtime.h"

namespace MAI {

class OpenCLRuntime : public Runtime {
public:
    enum class OpenCLDeviceType {
        UNKNOWN = -1,
        QCOM,
        MALI,
        POWER_VR,
    };
    enum class OpenCLVersionType {
        UNKNOWN = -1,
        VER_1_0,
        VER_1_1,
        VER_1_2,
        VER_2_0,
        VER_2_2,
    };
    OpenCLRuntime();
    ~OpenCLRuntime();
    OpenCLDeviceType deviceType();
    OpenCLVersionType versionType();
    cl::Context& context();
    cl::Device& device();
    cl::CommandQueue& commandQueue();
    bool buildKernel(
            const std::string& programName,
            const std::string& kernelName,
            const std::string& buildOptions,
            cl::Kernel& kernel);

    bool buildProgram(const std::string& programName,
            const std::string& buildOptions,
            cl::Program& program);

    bool buildProgramFromSource(const std::string& programName,
            const std::string& buildOptions,
            cl::Program& program);

    bool buildProgramFromBinary(const std::string& programName,
            const std::string& buildOptions,
            cl::Program& program);
private:
    std::shared_ptr<cl::Context> mContext;
    std::shared_ptr<cl::Device> mDevice;
    std::shared_ptr<cl::CommandQueue> mCommandQueue;
    // key is "programName + buildOptions"
    std::map<std::string, cl::Program> mProgramMap;
    OpenCLDeviceType mDeviceType;
    OpenCLVersionType mVersionType;
};

} // namespace MAI
