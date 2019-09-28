#include "TensorflowNetwork.h"
#include "source/util/MAIUtil.h"
#include "Tensor.h"

using namespace MAI;
int main() {
    uint64_t startTime, endTime, totalTime;
    MAI::TensorflowNetwork network("tools/converter/tensorflow/models/mobilenet-v1-1.0.pb",
            "tools/converter/tensorflow/protos/ops.pbtxt");
    startTime = nowMicros();
    network.init();
    endTime = nowMicros();
    ALOGI("Init time:%lu ms", (endTime - startTime) / 1000);
    const char* inputData = MAI::mapFile<char>("mobilenetv1_input.bin");
    MAI::Tensor* inputTensor = network.getTensor(network.getModelInputs()[0]);
    inputTensor->copy(inputData, inputTensor->size());
    startTime = nowMicros();
    network.run();
    endTime = nowMicros();
    ALOGI("First run time:%lu ms", (endTime - startTime) / 1000);
    int32 count = 1;
    int32 i = count;
    while(i-- > 0) {
        startTime = nowMicros();
        network.run();
        endTime = nowMicros();
        totalTime += (endTime - startTime);
        ALOGI("Run %d time:%lu ms", i, (endTime - startTime) / 1000);
    }
    ALOGI("Avg time:%lu", totalTime / count / 1000);
    return 0;
}
