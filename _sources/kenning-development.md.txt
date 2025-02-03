# Developing Kenning blocks

This chapter describes the development process of Kenning components.

To run below examples it is required to install Kenning with dependencies as follows:
```bash
pip install "kenning[tensorflow] @ git+https://github.com/antmicro/kenning.git"
```

(model-io-metadata)=
## Model and I/O metadata

Since not all model formats supported by Kenning provide information about inputs and outputs or other data required for compilation or runtime purposes, each model processed by Kenning comes with a JSON file describing an I/O specification (and other useful metadata).

The JSON file with metadata for file `<model-name>.<ext>` is saved as `<model-name>.<ext>.json` in the same directory.

The example metadata file looks as follows (for ResNet50 for the ImageNet classification problem):

```json
{
    "input": [
        {
            "name": "input_0",
            "shape": [1, 224, 224, 3],
            "dtype": "int8",
            "order": 0,
            "scale": 1.0774157047271729,
            "zero_point": -13,
            "prequantized_dtype": "float32"
        }
    ],
    "output": [
        {
            "name": "output_0",
            "shape": [1, 1000],
            "dtype": "int8",
            "order": 0,
            "scale": 0.00390625,
            "zero_point": -128,
            "prequantized_dtype": "float32"
        }
    ]
}
```

A sample metadata JSON file may look as follows (for the YOLOv4 detection model):

```json
{
    "input": [
        {
            "name": "input",
            "shape": [1, 3, 608, 608],
            "dtype": "float32"
        }
    ],
    "output": [
        {
            "name": "output",
            "shape": [1, 255, 76, 76],
            "dtype": "float32"
        },
        {
            "name": "output.3",
            "shape": [1, 255, 38, 38],
            "dtype": "float32"
        },
        {
            "name": "output.7",
            "shape": [1, 255, 19, 19],
            "dtype": "float32"
        }
    ],
    "processed_output": [
        {
            "name": "detection_output",
            "type": "List",
            "dtype": {
                "type": "List",
                "dtype": "kenning.datasets.helpers.detection_and_segmentation.DetectObject"
            }
        }
    ],
    "model_version": 4
}
```

In general, the metadata file consist of four fields:

* `input` is a specification of input data received by model wrapper (before preprocessing),
* `processed_input` is a specification of data passed to wrapped model (after preprocessing),
* `output` is a specification of data returned by wrapped model (before postprocessing),
* `processed_output` is a specification of output data returned by model wrapper (after postprocessing).

If `processed_input` or `processed_output` is not specified we assume that there is no processing and it is the same as `input` or `output` respectively.
Each array consist of dictionaries describing model inputs and outputs.

Additionally, any non-conflicting miscellaneous keywords may be included into the metadata JSON, but those will not be validated by Kenning.

Parameters common to all fields (except for the miscellaneous):

* `name` - input/output name,
* `shape` - input/output tensor shape, if `processed_input` is present, `input` can have list of valid shapes,
* `dtype` - input/output type,

Parameters specific to `processed_input` and `output`:

* `order` - some of the runtimes/optimizers allow accessing inputs and outputs by id.
  This field describes the id of the current input/output,
* `scale` - scale parameter for the quantization purposes.
  Present only if the input/output requires quantization/dequantization,
* `zero_point` - zero point parameter for the quantization purposes.
  Present only if the input/output requires quantization/dequantization,
* `prequantized_dtype` - input/output data type before quantization.
* `class_name` - list of class names from the dataset.
  It is used by output collectors to present the data in a human-readable way.

Parameters specific to `input`:

* `mean` - mean used to normalize inputs before model training,
* `std` - standard deviation used to normalize inputs before model training.

Parameters specific to `output` and `processed_output`:

* `class_name` - list of class names from the dataset.
  It is used by output collectors to present the data in a human-readable way.

Parameters specific to all fields which data is not compatible with `np.ndarray`:
* `type` - can be either `List`, `Dict` or `Any`.
* `dtype` - if `type` is `List`, it can specify type of elements, should be represented as whole path which can be imported (i.e. `kenning.datasets.helpers.detection_and_segmentation.SegmObject` for segmentation results).
It can also receive dictionary with specification of list items.
* `fields` - if `type` is `Dict`, it should be a dictionary with names as keys and values as dictionary matching `input` and `output` specification.
Example from `PyTorchCOCOMaskRCNN`:

```json
{
    "type": "Dict",
    "fields": {
        "boxes": {"shape": [-1, 4], "dtype": "float32"},
        "labels": {"shape": [-1,], "dtype": "int64"},
        "scores": {"shape": [-1,], "dtype": "float32"},
        "masks": {
            "shape": [-1, 1, 416, 416],
            "dtype": "float32",
        },
    },
}
```

The model metadata is used by all classes in Kenning in order to understand the format of the inputs and outputs.

```{warning}
The [](optimizer-api) objects can affect the format of inputs and outputs, e.g. quantize the network. It is crucial to update the I/O specification when a block modifies it.
```

## Implementing a new Kenning component

Firstly, check [](kenning-api) for available building blocks for Kenning and their documentation.
Adding a new block to Kenning is a matter of creating a new class inheriting from one of `kenning.core` classes, providing configuration parameters, and implementing methods - at least the unimplemented ones.

For example purposes, let's create a sample [](optimizer-api)-based class, which will convert an input model to the TensorFlow Lite format.

First, let's create minimal code with an empty class:

```python
from kenning.core.optimizer import Optimizer


class TensorFlowLiteCompiler(Optimizer):
    pass
```

(defining-arguments-for-core-classes)=

### Defining arguments for core classes

Kenning classes can be created in three ways:

* Using constructor in Python,
* Using argparse from command-line (see [](cmd-usage)),
* Using JSON-based dictionaries from JSON scenarios (see [](json-scenarios))

To support all three methods, the newly implemented class requires creating a dictionary called `arguments_structure` that holds all configurable parameters of the class, along with their description, type and additional information.

This structure is used to create:

* an `argparse` group, to configure class parameters from terminal level (via the [](argumentshandler-api)'s `form_argparse` method).
  Later, a class can be created with the `from_argparse` method.
* a JSON schema to configure the class from a JSON file (via the [](argumentshandler-api)'s `form_parameterschema` method).
  Later, a class can be created with the `from_parameterschema` method.

`arguments_structure` is a dictionary in the following form:

```python
arguments_structure = {
    'argument_name': {
        'description': 'Help for the argument',
        'type': str,
        'required': True
    }
}
```

The `argument_name` is a name used in:

* the Python constructor,
* an Argparse argument (in a form of `--argument-name`),
* a JSON argument.

The fields describing the argument are as follows:

* `argparse_name` - if there is a need for a different flag in argparse, it can be provided here,
* `description` - description of the node, displayed for a parsing error for JSON, or in help in case of command-line access,
* `type` - type of argument, i.e.:
    * `Path` from `pathlib` module,
    * `str`,
    * `float`,
    * `int`,
    * `bool`,
* `default` - default value for the argument,
* `required` - boolean, tells if argument is required or not,
* `enum` - a list of possible values for the argument,
* `is_list` - tells if argument is a list of objects of types given in `type` field,
* `nullable` - tells if argument can be empty (`None`).

Let's add parameters to the example class:

```python
from pathlib import Path
from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset


class TensorFlowLiteCompiler(Optimizer):
    arguments_structure = {
        'inferenceinputtype': {
            'argparse_name': '--inference-input-type',
            'description': 'Data type of the input layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'inferenceoutputtype': {
            'argparse_name': '--inference-output-type',
            'description': 'Data type of the output layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'quantize_model': {
            'argparse_name': '--quantize-model',
            'description': 'Tells if model should be quantized',
            'type': bool,
            'default': False
        },
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to '
                           '1.0) will be used for calibration dataset',
            'type': float,
            'default': 0.25
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantize_model: bool = False):
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.dataset_percentage = dataset_percentage
        self.quantize_model = quantize_model
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantize_model
        )
```

In addition to defined arguments, there are also default [](optimizer-api) arguments - the [](dataset-api) object and path to save the model (`compiled_model_path`).

Also, a `from_argparse` object creator is implemented, since there are additional parameters (`dataset`) to handle. The `from_parameterschema` function is created automatically.

Additionally, if the new class does not inherit from any Kenning core classes, then it should inherit directly from [](argumentshandler-api) class which is responsible for creating `argparse` groups and JSON schema from `arguments_structure`.

The above implementation of arguments is common for all core classes.

### Defining supported output and input types

The Kenning classes for consecutive steps are meant to work in a seamless manner, which means providing various ways to pass the model from one class to another.

Usually, each class can accept multiple model input formats and provides at least one output format that can be accepted in other classes (except for `terminal` compilers, such as [Apache TVM](https://tvm.apache.org/), that compile models to a runtime library).

The list of supported output formats is represented in a class with an `outputtypes` list:

```python test-skip
    outputtypes = [
        'tflite'
    ]
```

The supported input formats are delivered in a form of a dictionary, mapping the supported input type name to the function used to load a model:

```python test-skip
    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion
    }
```

Let's update the code with supported types:

```python
from kenning.core.optimizer import Optimizer
import tensorflow as tf
from pathlib import Path


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter


def tensorflowconversion(modelpath: Path):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelpath)
    return converter


class TensorFlowLiteCompiler(Optimizer):
    arguments_structure = {
        'inferenceinputtype': {
            'argparse_name': '--inference-input-type',
            'description': 'Data type of the input layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'inferenceoutputtype': {
            'argparse_name': '--inference-output-type',
            'description': 'Data type of the output layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'quantize_model': {
            'argparse_name': '--quantize-model',
            'description': 'Tells if model should be quantized',
            'type': bool,
            'default': False
        },
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to '
                           '1.0) will be used for calibration dataset',
            'type': float,
            'default': 0.25
        }
    }

    outputtypes = [
        'tflite'
    ]

    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantize_model: bool = False):
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.dataset_percentage = dataset_percentage
        self.quantize_model = quantize_model
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantize_model
        )
```

### Implementing unimplemented methods

The remaining aspect of developing Kenning classes is implementing unimplemented methods.
Unimplemented methods raise the `NotImplementedError`.

In case of the [](optimizer-api) class, it is the `compile` method and `get_framework_and_version`.

When implementing unimplemented methods, it is crucial to follow type hints both for inputs and outputs - more details can be found in the documentation for the method.
Sticking to the type hints ensures compatibility between blocks, which is required for a seamless connection between compilation components.

Let's finish the implementation of the [](optimizer-api)-based class:

```python
from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, List


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter


def tensorflowconversion(modelpath: Path):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelpath)
    return converter


class TensorFlowLiteCompiler(Optimizer):
    arguments_structure = {
        'inferenceinputtype': {
            'argparse_name': '--inference-input-type',
            'description': 'Data type of the input layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'inferenceoutputtype': {
            'argparse_name': '--inference-output-type',
            'description': 'Data type of the output layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'quantize_model': {
            'argparse_name': '--quantize-model',
            'description': 'Tells if model should be quantized',
            'type': bool,
            'default': False
        },
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to '
                           '1.0) will be used for calibration dataset',
            'type': float,
            'default': 0.25
        }
    }

    outputtypes = [
        'tflite'
    ]

    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantize_model: bool = False):
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.dataset_percentage = dataset_percentage
        self.quantize_model = quantize_model
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantize_model
        )

    def compile(
            self,
            inputmodelpath: Path,
            io_spec: Optional[Dict[str, List[Dict]]] = None):

        # load I/O specification for the model
        if io_spec is None:
            io_spec = self.load_io_specification(inputmodelpath)

        # load the model using chosen input type
        converter = self.inputtypes[self.inputtype](inputmodelpath)

        # preparing model compilation using class arguments
        if self.quantize_model:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        converter.inference_input_type = tf.as_dtype(self.inferenceinputtype)
        converter.inference_output_type = tf.as_dtype(self.inferenceoutputtype)

        # dataset can be used during compilation i.e. for calibration
        # purposes
        if self.dataset and self.quantize_model:
            def generator():
                for entry in self.dataset.calibration_dataset_generator(
                        self.dataset_percentage):
                    yield [np.array(entry, dtype=np.float32)]
            converter.representative_dataset = generator

        # compile and save the model
        tflite_model = converter.convert()
        with open(self.compiled_model_path, 'wb') as f:
            f.write(tflite_model)

        # update the I/O specification
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signature = interpreter.get_signature_runner()

        def update_io_spec(sig_det, int_det, key):
            for order, spec in enumerate(io_spec[key]):
                old_name = spec['name']
                new_name = sig_det[old_name]['name']
                spec['name'] = new_name
                spec['order'] = order

            quantized = any([det['quantization'][0] != 0 for det in int_det])
            new_spec = []
            for det in int_det:
                spec = [
                    spec for spec in io_spec[key]
                    if det['name'] == spec['name']
                ][0]

                if quantized:
                    scale, zero_point = det['quantization']
                    spec['scale'] = scale
                    spec['zero_point'] = zero_point
                    spec['prequantized_dtype'] = spec['dtype']
                    spec['dtype'] = np.dtype(det['dtype']).name
                new_spec.append(spec)
            io_spec[key] = new_spec

        update_io_spec(
            signature.get_input_details(),
            interpreter.get_input_details(),
            'input'
        )
        update_io_spec(
            signature.get_output_details(),
            interpreter.get_output_details(),
            'output'
        )

        # save updated I/O specification
        self.save_io_specification(inputmodelpath, io_spec)

    def get_framework_and_version(self):
        return 'tensorflow', tf.__version__
```

There are several important things regarding the code snippet above:

* The information regarding inputs and outputs can be collected with the `self.load_io_specification` method, present in all classes.
* The information about the input format to use is delivered in the `self.inputtype` field - it is updated automatically by the function consulting the best supported format for previous and current block.
* If the I/O metadata is affected by the current block, it needs to be updated and saved along with the compiled model using the `self.save_io_specification` method.

### Using the implemented block

In the Python script, the above class can be used with other classes from the [](kenning-api) as is - it is a regular Python code.

To use the implemented block in the JSON scenario (as described in [](json-scenarios), the module implementing the class needs to be available from the current directory, or the path to the module needs to be added to the `PYTHONPATH` variable.

Let's assume that the class was implemented in the `my_optimizer.py` file.
The scenario can look as follows:

{ emphasize-lines="18-29" }
```json
{
    "model_wrapper":
    {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters":
        {
            "model_path": "kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
        }
    },
    "dataset":
    {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters":
        {
            "dataset_root": "./build/pet-dataset"
        }
    },
    "optimizers":
    [
        {
            "type": "my_optimizer.TensorFlowLiteCompiler",
            "parameters":
            {
                "compiled_model_path": "./build/compiled-model.tflite",
                "inference_input_type": "float32",
                "inference_output_type": "float32"
            }
        }
    ],
    "runtime":
    {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters":
        {
            "save_model_path": "./build/compiled-model.tflite"
        }
    }
}
```

The emphasized line demonstrates usage of the implemented `TensorFlowLiteCompiler` from the `my_optimizer.py` script.

This sums up the Kenning development process.

## Implementing Kenning runtime blocks

(implementing-runner)=
### Implementing new Runners for KenningFlow

The process of creating new [](runner-api) is almost the same as the process of implementing Kenning components described above, with a few additional steps.

First of all, the new component needs to inherit from a Runner class (not necessarily directly).
Then you need to implement the following methods:
* `cleanup` - cleans resources after execution is stopped
* `should_close` - (*optional*) returns boolean indicating whether a runner ended processing and requests closing
  Possible reasons: a signal from terminal, user request in GUI, end of data to process, an error, and more.
  Default implementation always returns `False`
* `run` - method that gets runner inputs, processes them and returns obtained results.

Inputs for the runner are passed to the `run` method as a dictionary where the key is the input name specified in the KenningFlow JSON and the value is simply the input value.

E.g. for `DetectionVisualizer` defined in JSON as

{ emphasize-lines="8-9" }
```json
{
    "type": "kenning.outputcollectors.detection_visualizer.DetectionVisualizer",
    "parameters": {
        "output_width": 608,
        "output_height": 608
    },
    "inputs": {
        "frame": "cam_frame",
        "detection_data": "predictions"
    }
}
```

the `run` method access inputs as follows

{ emphasize-lines="4-5" }
```python
from typing import Any, Dict

def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    input_data = inputs['frame']
    output_data = inputs['detection_data']
    self.process_output(input_data, output_data)
```

```{note}
In the example above, the `run` method does not contain a `return` statement, because this runner does not have any outputs.
If you want to create a runner with outputs, this method should return a similar dictionary containing outputs.
```
