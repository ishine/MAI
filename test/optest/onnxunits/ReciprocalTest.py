import onnx
from onnx import helper
from onnx import TensorProto
import caffe2.python.onnx.backend as backend
import numpy as np

node=helper.make_node('Reciprocal', ['x'], ['y'])

graph = helper.make_graph([node], 'reciprocal-test', [helper.make_tensor_value_info('x', TensorProto.FLOAT, (1,5))],
         [helper.make_tensor_value_info('y', TensorProto.FLOAT, (1,5))],)

model= helper.make_model(graph)

onnx.checker.check_model(model)

rep = backend.prepare(model, device='CPU')
X=np.array([-2.0, -1, 1, 2], dtype=np.float32)
ret = rep.run(X)
print ret
