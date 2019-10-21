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

model = onnx.load("squeezenet_opset_9.onnx")
addModelOutput(model, "fire2/concat_1", [1,128,55,55])
#addModelOutput(model, "pool10_1", [1,1000,1,1])
#addModelOutput(model, "fc7_2", [1,4096])

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
index+=1;outputs[index].tofile(file="output/softmaxout_1.txt", sep="\n")
index+=1;outputs[index].tofile(file="output/fire2_concat_1.txt", sep="\n")
#index+=1;outputs[index].tofile(file="output/pool10_1.txt", sep="\n")
#index+=1;outputs[index].tofile(file="output/fc7_2.txt", sep="\n")
#index+=1;outputs[index].tofile(file="output/1137.txt", sep="\n")
