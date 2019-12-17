# MAI

MAI是一个神经网络推理引擎，以跨平台为目标，开发目标：Linux CPU --> Android CPU --> Android GPU --> Android DSP --> Windows CPU -->...

## Benchmark

./bazel-bin/tools/benchmark/mai_benchmark --model_format=TENSORFLOW --model_path=tools/converter/tensorflow/models/mobilenet-v1-1.0.pb --num_runs=1 --warm_up=0

## Supported models

| Models                            | ONNX     | Tensorflow |
| ---                               | ---      | ---      |            
|mobilenetv1                        | yes      |   yes    |        
|squeezenet                        | [yes](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz)     |   no(https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz)    |
|shufflenet                        | [yes](https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz)     |   no    |

## Supported operators
<table>
   <tr>
      <td align="center" rowspan=2><h4>Operators</h4></td>
      <td align="center" colspan=2><h4>Linux</h4></td>
      <td align="center" colspan=2><h4>Windows</h4></td>
      <td align="center" colspan=4><h4>Android</h4></td>
   </tr>
   <tr>
      <td>CPU</td>
      <td>GPU</td>
      <td>CPU</td>
      <td>GPU</td>
      <td>CPU</td>
      <td>GPU</td>
      <td>DSP</td>
      <td>NPU</td>
   </tr>
   <tr>
      <td>Add</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>BiasAdd</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Cast</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Concat</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Conv2D</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>DepthwiseConv2d</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>TransposeConv2D</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Dropout</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ExpandDims</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Exp</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Fill</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Floor</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>FusedBatchNorm</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
  <tr>
      <td>Gather</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Gemm</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Mul</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Pad</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>MaxPool</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>AveragePool</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>GlobalAveragePool</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Relu</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Relu1</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Relu6</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>CRelu</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>LeakyRelu</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Reshape</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Shape</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Sigmoid</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Softmax</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Split</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Squeeze</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Tan</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Tanh</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ATan</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ATanh</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Sin</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Sinh</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ASin</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ASinh</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Cos</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Cosh</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ACos</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ACosh</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Transpose</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>StridedSlice</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Identity</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Pack</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ResizeBilinear</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Pow</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Neg</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ArgMax</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>ArgMin</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
   <tr>
      <td>Abs</td> <td>yes</td> <td></td> <td></td> <td></td> <td>yes</td> <td></td> <td></td> <td></td>
   </tr>
</table>
