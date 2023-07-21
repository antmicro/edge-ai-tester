# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for ONNX deep learning compiler.
"""

import onnx
from typing import Optional, Dict, List

from kenning.core.dataset import Dataset
from kenning.core.optimizer import Optimizer
from kenning.core.optimizer import ConversionError
from kenning.core.optimizer import CompilationError
from kenning.core.optimizer import IOSpecificationNotFoundError
from kenning.utils.resource_manager import PathOrURI, ResourceURI


def kerasconversion(model_path: PathOrURI, input_spec, output_names):
    import tensorflow as tf
    import tf2onnx
    model = tf.keras.models.load_model(str(model_path))

    input_spec = [tf.TensorSpec(
        spec['shape'],
        spec['dtype'],
        name=spec['name']
    ) for spec in input_spec]
    modelproto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec
    )

    return modelproto


def torchconversion(model_path: PathOrURI, input_spec, output_names):
    import torch
    dev = 'cpu'
    model = torch.load(str(model_path), map_location=dev)

    if not isinstance(model, torch.nn.Module):
        raise CompilationError(
            f'ONNX compiler expects the input data of type: torch.nn.Module, but got: {type(model).__name__}'  # noqa: E501
        )

    model.eval()

    input = tuple(torch.randn(
        spec['shape'],
        device=dev
    ) for spec in input_spec)

    import io
    mem_buffer = io.BytesIO()
    torch.onnx.export(
        model,
        input,
        mem_buffer,
        opset_version=11,
        input_names=[spec['name'] for spec in input_spec],
        output_names=output_names
    )
    onnx_model = onnx.load_model_from_string(mem_buffer.getvalue())
    return onnx_model


def tfliteconversion(model_path: PathOrURI, input_spec, output_names):
    import tf2onnx
    try:
        modelproto, _ = tf2onnx.convert.from_tflite(
            str(model_path)
        )
    except ValueError as e:
        raise ConversionError(e)

    return modelproto


class ONNXCompiler(Optimizer):
    """
    The ONNX compiler.
    """
    inputtypes = {
        'keras': kerasconversion,
        'torch': torchconversion,
        'tflite': tfliteconversion
    }

    outputtypes = ['onnx']

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys())
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: PathOrURI,
            modelframework: str = 'keras'):
        """
        The ONNX compiler.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        modelframework : str
            Framework of the input model, used to select a proper backend.
        """
        self.modelframework = modelframework
        self.set_input_type(modelframework)
        super().__init__(dataset, compiled_model_path)

    def compile(
            self,
            input_model_path: PathOrURI,
            io_spec: Optional[Dict[str, List[Dict]]] = None):
        input_model_path = ResourceURI(input_model_path)

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        try:
            from copy import deepcopy
            io_spec = deepcopy(io_spec)

            input_spec = io_spec['input']
            output_spec = io_spec['output']
        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError('No input/output specification found')  # noqa: E501

        try:
            output_names = [spec['name'] for spec in output_spec]
        except KeyError:
            output_names = None

        model = self.inputtypes[self.inputtype](
            input_model_path,
            input_spec,
            output_names
        )

        onnx.save(model, self.compiled_model_path)

        # update the io specification with names
        for spec, input in zip(input_spec, model.graph.input):
            spec['name'] = input.name

        for spec, output in zip(output_spec, model.graph.output):
            spec['name'] = output.name

        self.save_io_specification(
            input_model_path,
            {
                'input': input_spec,
                'output': output_spec
            }
        )
        return 0

    def get_framework_and_version(self):
        return ('onnx', onnx.__version__)
