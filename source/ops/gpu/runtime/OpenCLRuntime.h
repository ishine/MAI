// Copyright 2019 MAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include "CL/cl2.hpp"
#include "include/Runtime.h"

namespace MAI {

class CLPrecompiledFile {
public:
    CLPrecompiledFile(const std::string& file) : mFile(file) {}
    MAI_STATUS load();
    MAI_STATUS store();
    void add(const std::string& key, const std::vector<uint8>& value);
    std::vector<uint8>* find(const std::string& key);
private:
    std::string mFile;
    std::map<std::string, std::vector<uint8>> mData;
};

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
    static std::string openCLErrorCodeToString(cl_int err);
    OpenCLRuntime();
    ~OpenCLRuntime();
    OpenCLDeviceType deviceType();
    OpenCLVersionType versionType();
    cl::Context& context();
    cl::Device& device();
    cl::CommandQueue& commandQueue();
    void setCLDir(const std::string& dir);
    std::string getCLDir();
    bool buildKernel(
            const std::string& programName,
            const std::string& kernelName,
            const std::string& buildOptions,
            cl::Kernel& kernel);

    bool buildProgram(
            const std::string& programName,
            const std::string& buildOptions,
            cl::Program& program);

    bool buildProgramFromSource(
            const std::string& programKey,
            const std::string& programName,
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
    std::string mCLDir;
    CLPrecompiledFile mPrecompiledFile;
};

} // namespace MAI
