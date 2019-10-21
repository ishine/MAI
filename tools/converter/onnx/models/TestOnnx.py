import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

model = onnx.load("models/mobilenet_v1_1.0_224.onnx")
print(type(model))

onnx.checker.check_model(model)

onnx.helper.printable_graph(model.graph)

rep = backend.prepare(model, device="CPU")
print(type(rep))
#input=np.random.randn(1, 3, 224, 224)
#input=np.fromfile("models/mobilenetv1_input.bin")
input=np.fromfile("input.txt", sep='\n').reshape([1,3,224,224])
input.tofile(file="input.data", sep="\n")
outputs = rep.run(input.astype(np.float32))
#outputs[0].tofile(file="output.data", sep="\n")
print(outputs[0])
