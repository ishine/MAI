#include "TensorflowNetwork.h"

int main() {
    MAI::TensorflowNetwork network("tools/converter/tensorflow/models/mobilenet-v1-1.0.pb",
            "tools/converter/tensorflow/protos/ops.pbtxt");
    return 0;
}
