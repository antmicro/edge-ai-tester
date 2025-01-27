# Defining optimization pipelines in Kenning

{{project}} blocks (specified in the [](kenning-api)) can be configured either via command line (see [](cmd-usage)), or via configuration files, specified in JSON format.
The latter approach allows the user to create more advanced and easy-to-reproduce scenarios for model deployment.
Most notably, various optimizers available through Kenning can be chained to utilize various optimizations and get better performing models.

One of the scenarios most commonly used in Kenning is model optimization and compilation.
It can be done using {{json_compilation_script}}.

To run below examples it is required to install Kenning with dependencies as follows:
```bash
pip install "kenning[tensorflow,tflite,tvm] @ git+https://github.com/antmicro/kenning.git"
```

## JSON specification

The {{json_compilation_script}} takes the specification of optimization and the testing flow in a JSON format.
The root element of the JSON file is a dictionary that can have the following keys:

* `model_wrapper` - **mandatory field**, accepts dictionary as a value that defines the [](modelwrapper-api) object for the deployed model (provides I/O processing, optionally model).
* `dataset` - **mandatory field**, accepts dictionary as a value that defines the [](dataset-api) object for model optimization and evaluation.
* `optimizers` - *optional field*, accepts a list of dictionaries specifying the sequence of [](optimizer-api)-based optimizations applied to the model.
* `protocol` - *optional field*, defines the [](protocol-api) object used to communicate with a remote target platform.
* `runtime` - *optional field* (**required** when `optimizers` are provided), defines the [](runtime-api)-based object that will infer the model on target device.

Each dictionary in the fields above consists of:

* `type` - appropriate class for the key,
* `parameters` - `type`-specific arguments for an underlying class (see [](defining-arguments-for-core-classes)).

## Model evaluation using its native framework

The simplest JSON configuration looks as follows:

```{literalinclude} scripts/jsonconfigs/mobilenetv2-tensorflow-native.json save-as=mobilenetv2-tensorflow-native.json
:language: json
```

It only takes `model_wrapper` and `dataset`.
This way, the model will be loaded and evaluated using its native framework.

The [](modelwrapper-api) used is `TensorFlowPetDatasetMobileNetV2`, which is a MobileNetV2 model trained to classify 37 breeds of cats and dogs.
In the `type` field, we specify the full "path" to the class by specifying the module it is implemented in (`kenning.modelwrappers.classification.tensorflow_pet_dataset`) and the name of the class (`TensorFlowPetDatasetMobileNetV2`) in a Python-like format (dot-separated).

In `parameters`, arguments specific to `TensorFlowPetDatasetMobileNetV2` are provided.
The following parameters are available based on the argument specification:

```python test-skip
# this argument structure is taken from kenning.core.model - it is inherited by child classes
arguments_structure = {
    'modelpath': {
        'argparse_name': '--model-path',
        'description': 'Path to the model',
        'type': Path,
        'required': True
    }
}
```

The only mandatory parameter here is `model_path`, which points to a file containing the model.
It is a required argument.

The `dataset` used here, is `PetDataset`. Like previously, it is provided in a module-like format (`kenning.datasets.pet_dataset.PetDataset`). The parameters here are specified in `kenning.core.dataset.Dataset` (inherited) and `kenning.core.dataset.PetDataset`:

```python test-skip
arguments_structure = {
    # coming from kenning.core.dataset.Dataset
    'root': {
        'argparse_name': '--dataset-root',
        'description': 'Path to the dataset directory',
        'type': Path,
        'required': True
    },
    'batch_size': {
        'argparse_name': '--inference-batch-size',
        'description': 'The batch size for providing the input data',
        'type': int,
        'default': 1
    },
    'download_dataset': {
        'description': 'Downloads the dataset before taking any action',
        'type': bool,
        'default': False
    },
    # coming from kenning.datasets.pet_dataset.PetDataset
    'classify_by': {
        'argparse_name': '--classify-by',
        'description': 'Determines if classification should be performed by species or by breeds',
        'default': 'breeds',
        'enum': ['species', 'breeds']
    },
    'image_memory_layout': {
        'argparse_name': '--image-memory-layout',
        'description': 'Determines if images should be delivered in NHWC or NCHW format',
        'default': 'NHWC',
        'enum': ['NHWC', 'NCHW']
    }
}
```

As visible, the parameters allow the user to:

* specify the dataset's location,
* download the dataset,
* configure data layout and batch size,
* configure anything specific to the dataset.

```{note}
For more details on defining parameters for Kenning core classes, check [](defining-arguments-for-core-classes).
```

If `optimizers` or `runtime` are not specified, the model is executed using the [](modelwrapper-api)'s `run_inference` method.
The dataset test data is passed through the model and evaluation metrics are collected.

To run the defined pipeline (assuming that the JSON file is under `pipeline.json`), run:

```bash timeout=10
kenning test \
    --json-cfg mobilenetv2-tensorflow-native.json \
    --measurements measurements.json \
    --verbosity INFO
```

The `measurements.json` file is the output of the {{json_compilation_script}} providing measurement data.
It contains information such as:

* the JSON configuration defined above,
* versions of core class packages used (e.g. `tensorflow`, `torch`, `tvm`),
* available resource usage readings (CPU usage, GPU usage, memory usage),
* data necessary for evaluation, such as predictions, confusion matrix, etc.

This information can be later used for [](report-generation).

```{note}
Check {doc}`kenning-measurements` for more information.
```

## Optimizing and running a model on a single device

Model optimization and deployment can be performed directly on target device, if the device is able to perform the optimization steps.
It can also be used to check the outcome of certain optimizations on a desktop platform before deployment.

Optimizations and compilers used in a scenario are defined in the `optimizers` field.
This field accepts a list of optimizers - they are applied to the model in the same order in which they are defined in the `optimizers` field.

For example, a model can be subjected to the following optimizations:

* Quantization of weights and activations using TensorFlow Lite.
* Conversion of data layout from NHWC to NCHW format using Apache TVM
* Compilation to x86 runtime with AVX2 vector extensions using Apache TVM.

Such case will result is the following scenario:

```{literalinclude} scripts/jsonconfigs/mobilenetv2-tensorflow-tvm-avx-int8.json save-as=mobilenetv2-tensorflow-tvm-avx-int8.json
:language: json
:emphasize-lines: 15-40
```

As emphasized above, the `optimizers` list is added, with two entries:

* a `kenning.optimizers.tflite.TFLiteCompiler` type block, quantizing the model,
* a `kenning.optimizers.tvm.TVMCompiler` type block, performing remaining optimization steps.

In the `runtime` field, a TVM-specific `kenning.runtimes.tvm.TVMRuntime` type is used.

The first optimizer on the list reads the input model path from the [](modelwrapper-api)'s `model_path` field.
Each consecutive [](optimizer-api) reads the model from a file saved by the previous [](optimizer-api).
In the simplest scenario, the model is saved to `compiled_model_path` in each optimizer, and is fetched by the next [](optimizer-api).

In case the default output file type of the previous [](optimizer-api) is not supported by the next [](optimizer-api), the first common supported model format is determined and used to pass the model between optimizers.

In case no such format exists, the {{json_compilation_script}} returns an error.

```{note}
More details on input/output formats between [](optimizer-api) objects can be found in [](kenning-development).
```

The scenario can be executed as follows:

```bash timeout=20
kenning optimize test --json-cfg mobilenetv2-tensorflow-tvm-avx-int8.json --measurements output.json
```

## Compiling a model and running it remotely

For some platforms, we cannot run a Python script to evaluate or run the model to check its quality - the dataset is too large to fit in the storage, no libraries or compilation tools are available for the target platform, or the device does not have an operating system to run Python on.

In such cases, it is possible to evaluate the system remotely using the [](protocol-api) and the ``kenning.scenarios.inference_server`` scenario.

For this use case, we need two JSON files - one for inference server configuration, and another one for the ``kenning.scenarios.inference_tester`` configuration, which acts as a runtime client.

The client and the server may communicate via different means, protocols and interfaces - we can use TCP communication, UART communication or other.
It depends on the [](protocol-api) used.

In addition, in such scenario optimizers can be executed either on host (which is default behavior) or on target device.
To specify it, you can use `location` parameter of the [](optimizer-api).

To create client/server scenario configuration it is required to add a `protocol` entry:

```{literalinclude} scripts/jsonconfigs/tflite-tvm-classification-client-server.json save-as=tflite-tvm-classification-client-server.json
:language: json
:emphasize-lines: 48-57
```

In the `protocol` entry, we specify a `kenning.protocols.network.NetworkProtocol` and provide a server address (`host`), an application port (`port`) and packet size (`packet_size`)

The server parses only `runtime` and `protocol` from the configuration, so any changes to the other of the blocks does not require server restart.
The server uses `protocol` to receive requests from clients and `runtime` to run the tested models.

The remaining things are provided by the client - input data and model.
Direct outputs from the model are sent as is to the client, so it can postprocess them and evaluate the model using the dataset.
The server also sends measurements from its sensors in JSON format as long as it is able to collect and send them.

First, run the server, so that it is available for the client:

```bash terminal=1
kenning server \
    --json-cfg tflite-tvm-classification-client-server.json \
    --verbosity INFO &
```

Then, run the client:

```bash timeout=60
kenning optimize test \
    --json-cfg tflite-tvm-classification-client-server.json \
    --measurements ./build/tflite-tvm-classification.json \
    --verbosity INFO
```

The rest of the flow is automated.

To execute one of the optimizers on the target-side, simply add the `location` parameter as follows:

{ emphasize-lines="20" }
```json
"optimizers":
[
    {
        "type": "kenning.optimizers.tflite.TFLiteCompiler",
        "parameters":
        {
            "target": "int8",
            "compiled_model_path": "./build/int8.tflite",
            "inference_input_type": "int8",
            "inference_output_type": "int8"
        }
    },
    {
        "type": "kenning.optimizers.tvm.TVMCompiler",
        "parameters": {
            "target": "llvm -mcpu=core-avx2",
            "opt_level": 3,
            "conv2d_data_layout": "NCHW",
            "compiled_model_path": "./build/int8_tvm.tar",
            "location": "target"
        }
    }
],
```

and start the client the same as above (it is not required to restart server).
