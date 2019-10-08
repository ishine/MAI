# MAI

MAI是一个神经网络推理引擎，以跨平台为目标，开发目标：Linux CPU --> Android CPU --> Android GPU --> Android DSP --> Windows CPU -->...

## Benchmark

./bazel-bin/tools/benchmark/mai_benchmark --model_format=TENSORFLOW --model_path=tools/converter/tensorflow/models/mobilenet-v1-1.0.pb --num_runs=1 --warm_up=0
