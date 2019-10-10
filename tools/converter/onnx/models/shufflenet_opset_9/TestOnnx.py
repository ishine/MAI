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

model = onnx.load("shufflenet_opset_9.onnx")
addModelOutput(model, "gpu_0/conv3_0_1", [1,24,112,112])
#addModelOutput(model, "gpu_0/conv3_0_bn_1", [1,24,112,112])
addModelOutput(model, "gpu_0/conv3_0_bn_2", [1,24,112,112])
addModelOutput(model, "gpu_0/pool_0_1", [1,24,56,56])
addModelOutput(model, "gpu_0/gconv1_0_1", [1,112,56,56])
addModelOutput(model, "gpu_0/gconv3_0_1", [1,112,28,28])
addModelOutput(model, "gpu_0/block0_1", [1,136,28,28])
addModelOutput(model, "gpu_0/gconv3_0_bn_1", [1,112,28,28])
#addModelOutput(model, "gpu_0/gconv1_1_bn_1", [1,112,28,28])
addModelOutput(model, "gpu_0/gconv1_0_bn_2", [1,112,56,56])
addModelOutput(model, "gpu_0/gconv1_7_bn_1", [1,136,28,28])
addModelOutput(model, "gpu_0/gconv1_3_bn_1", [1,136,28,28])

print(len(model.graph.output))
#data = model.SerializeToString();
#file=open("mobilenet_v1_1.0_224_all_outputs.onnx", "wb")
#file.write(data)

#onnx.checker.check_model(model)

#onnx.helper.printable_graph(model.graph)

rep = backend.prepare(model, device="CPU")
print(type(rep))
input=np.fromfile("input.txt", sep='\n').reshape([1,3,224,224])
#input.tofile(file="output/input.data", sep="\n")
outputs = rep.run(input.astype(np.float32))
index=-1
#index+=1;outputs[index].tofile(file="output/gpu_0_softmax_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_softmax_1.txt", sep="\n", format="%f")
index+=1;outputs[index].tofile(file="output/gpu_0_conv3_0_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_conv3_0_bn_2.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_pool_0_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_gconv1_0_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_gconv3_0_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_block0_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_gconv3_0_bn_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_gconv1_0_bn_2.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_gconv1_7_bn_1.txt", sep="\n", format="%10.8e")
index+=1;outputs[index].tofile(file="output/gpu_0_gconv1_3_bn_1.txt", sep="\n", format="%10.8e")
