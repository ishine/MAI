import onnx
import caffe2.python.onnx.backend as backend
import numpy as np
import torch

def addModelOutput(model, name, dims, elem_type=1):
    value_info = model.graph.output.add()
    value_info.type.tensor_type.elem_type=elem_type
    value_info.name = name
    for tmpDim in dims:
        dim = value_info.type.tensor_type.shape.dim.add()
        dim.dim_value = tmpDim

model = onnx.load("../mobilenet_v1_1.0_224.onnx")
#model = onnx.load("mobilenet_v1_1.0_224_all_outputs.onnx")
#print(type(model))
#help(model)
print(model.graph.output[0])
print(len(model.graph.input))
#model.graph.ClearField('output')
#addModelOutput(model, "Conv__224:0", [1,32,112,112])
#addModelOutput(model, "MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0", [1,32,112,112])
#addModelOutput(model, "MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise:0", [1,32,112,112])
addModelOutput(model, "MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm:0", [1,32,112,112])

print(len(model.graph.output))
#data = model.SerializeToString();
#file=open("mobilenet_v1_1.0_224_all_outputs.onnx", "wb")
#file.write(data)

#onnx.checker.check_model(model)

#onnx.helper.printable_graph(model.graph)

rep = backend.prepare(model, device="CPU")
print(type(rep))
#input=np.random.randn(1, 3, 224, 224)
#input=np.fromfile("models/mobilenetv1_input.bin")
input=np.fromfile("input.txt", sep='\n').reshape([1,3,224,224])
input.tofile(file="input.data", sep="\n")
outputs = rep.run(input.astype(np.float32))
index=-1
index+=1;outputs[index].tofile(file="output/MobilenetV1_Predictions_Reshape_1:0.txt", sep="\n")
#outputs[index++].tofile(file="output/Conv__224:0.data", sep="\n")
#index+=1;outputs[index].tofile(file="output/MobilenetV1_MobilenetV1_Conv2d_0_Relu6:0.text", sep="\n")
#index+=1;outputs[index].tofile(file="output/MobilenetV1_MobilenetV1_Conv2d_1_depthwise_depthwise:0.txt", sep="\n")
index+=1;outputs[index].tofile(file="output/MobilenetV1_MobilenetV1_Conv2d_1_depthwise_BatchNorm_FusedBatchNorm:0.txt", sep="\n")
