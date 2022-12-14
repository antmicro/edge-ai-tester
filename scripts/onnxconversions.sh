#!/bin/bash

set -e

python -m kenning.scenarios.onnx_conversion \
    build/models-directory \
    build/onnx-support.md \
    --converters-list \
        kenning.onnxconverters.mxnet.MXNetONNXConversion \
        kenning.onnxconverters.pytorch.PyTorchONNXConversion \
        kenning.onnxconverters.tensorflow.TensorFlowONNXConversion

cat build/onnx-support.md
