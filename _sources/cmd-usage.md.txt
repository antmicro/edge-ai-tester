# Using Kenning via command line arguments

{{projecturl}} provides several scripts for training, compilation and benchmarking of deep learning models on various target hardware.
The executable scripts are present in the [kenning.scenarios module](https://github.com/antmicro/kenning/tree/main/kenning/scenarios).
Sample bash scripts using the scenarios are located in the [scripts directory in the repository](https://github.com/antmicro/kenning/tree/main/scripts).

Runnable scripts in scenarios require implemented classes to be provided from the `kenning.core` module to perform such actions as in-framework inference, model training, model compilation and model benchmarking on target.

To run below examples it is required to install Kenning with dependencies as follows:
```bash
pip install "kenning[tensorflow,tvm] @ git+https://github.com/antmicro/kenning.git"
```

## Command-line arguments for classes

Each class ([](dataset-api), [](modelwrapper-api), [](optimizer-api) and other) provided to the runnable scripts in scenarios can provide command-line arguments that configure the work of an object of the given class.

Each class in `kenning.core` implements `form_argparse` and `from_argparse` methods.
The former creates an `argparse` group for a given class with its parameters.
The latter takes the arguments parsed by `argparse` and returns the object of a class.

## Autocompletion for command line interface

{{project}} provides autocompletion for its command line interface.
This feature requires additional configuration to work properly, which can be done using `kenning completion` command.
Optionally, it can be configured as described in [argcomplete documentation](https://kislyuk.github.io/argcomplete/).

## Model training

`kenning.scenarios.model_training` performs model training using Kenning's [](modelwrapper-api) and [](dataset-api) objects.
To get the list of training parameters, select the model and training dataset to use (i.e. `TensorFlowPetDatasetMobileNetV2` model and `PetDataset` dataset) and run:

```bash
kenning train \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    -h
```

This will list the possible parameters that can be used to configure the dataset, the model, and the training parameters.
For the above call, the output is as follows:

```bash test-skip
common arguments:
  -h, --help            show this help message and exit
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Verbosity level

'train' arguments:
  --modelwrapper-cls MODELWRAPPER_CLS
                        ModelWrapper-based class with inference implementation to import
  --dataset-cls DATASET_CLS
                        Dataset-based class with dataset to import
  --batch-size BATCH_SIZE
                        The batch size for training
  --learning-rate LEARNING_RATE
                        The learning rate for training
  --num-epochs NUM_EPOCHS
                        Number of epochs to train for
  --logdir LOGDIR       Path to the training logs directory

ModelWrapper arguments:
  --model-path MODEL_PATH
                        Path to the model

PetDataset arguments:
  --classify-by {species,breeds}
                        Determines if classification should be performed by species or by breeds
  --image-memory-layout {NHWC,NCHW}
                        Determines if images should be delivered in NHWC or NCHW format

Dataset arguments:
  --dataset-root DATASET_ROOT
                        Path to the dataset directory
  --inference-batch-size INFERENCE_BATCH_SIZE
                        The batch size for providing the input data
  --download-dataset    Downloads the dataset before taking any action. If the dataset files are already downloaded and
                        the checksum is correct then they are not downloaded again. Is enabled by default.
  --force-download-dataset
                        Forces dataset download
  --external-calibration-dataset EXTERNAL_CALIBRATION_DATASET
                        Path to the directory with the external calibration dataset
  --split-fraction-test SPLIT_FRACTION_TEST
                        Default fraction of data to leave for model testing
  --split-fraction-val SPLIT_FRACTION_VAL
                        Default fraction of data to leave for model valdiation
  --split-seed SPLIT_SEED
                        Default seed used for dataset split
```

```{note}
The list of options depends on [](modelwrapper-api) and [](dataset-api).
```

At the end, the training can be configured as follows:

```bash timeout=5
kenning train \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --logdir build/logs \
    --dataset-root build/pet-dataset \
    --model-path build/trained-model.h5 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --num-epochs 50
```

This will train the model with a 0.0001 learning rate and batch size 32 for 50 epochs.
The trained model will be saved as `build/trained-model.h5`.

## In-framework inference performance measurements

The `kenning.scenarios.inference_performance` script runs inference on a given model in a framework it was trained on.
It requires you to provide:

* a [](modelwrapper-api)-based object wrapping the model to be tested,
* a [](dataset-api)-based object wrapping the dataset applicable to the model,
* a path to the output JSON file with performance and quality metrics gathered during inference by the [](measurements-api) object.

The example call for the method is as follows:

```bash timeout=5
kenning test \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements build/tensorflow_pet_dataset_mobilenetv2.json \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --dataset-root build/pet-dataset/
```

The script downloads the dataset to the `build/pet-dataset` directory, loads the `tensorflow_pet_dataset_mobilenetv2.h5` model, runs inference on all images from the dataset and collects performance and quality metrics throughout the run.
The performance data stored in the JSON file can be later rendered using [](report-generation).

## Testing inference on target hardware

The `kenning.scenarios.inference_tester` and `kenning.scenarios.inference_server` are used for inference testing on target hardware.
The `inference_tester` loads the dataset and the model, compiles the model and runs inference either locally or remotely using `inference_server`.

The `inference_server` receives the model and input data, and sends output data and statistics.

Both `inference_tester` and `inference_server` require [](runtime-api) to determine the model execution flow.
Both scripts communicate using the communication protocol implemented in the [](protocol-api).

At the end, the `inference_tester` returns the benchmark data in the form of a JSON file extracted from the [](measurements-api) object.

The `kenning.scenarios.inference_tester` requires:

* a [](modelwrapper-api)-based class that implements model loading, I/O processing and optionally model conversion to ONNX format,
* a [](runtime-api)-based class that implements data processing and the inference method for the compiled model on the target hardware,
* a [](dataset-api)-based class that implements data sample fetching and model evaluation,
* a path to the output JSON file with performance and quality metrics.

An [](optimizer-api)-based class can be provided to compile the model for a given target if needed.

Optionally, it requires a [](protocol-api)-based class when running remotely to communicate with the `kenning.scenarios.inference_server`.

To print the list of required arguments, run:

```bash
kenning optimize test \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --compiler-cls kenning.optimizers.tvm.TVMCompiler \
    --protocol-cls kenning.protocols.network.NetworkProtocol \
    -h
```

With the above classes, the help can look as follows:

```bash test-skip
common arguments:
  -h, --help            show this help message and exit
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Verbosity level
  --convert-to-onnx CONVERT_TO_ONNX
                        Before compiling the model, convert it to ONNX and use in compilation (provide a path to save here)
  --measurements MEASUREMENTS
                        The path to the output JSON file with measurements

Inference configuration with JSON:
  Configuration with pipeline defined in JSON file. This section is not compatible with 'Inference configuration with flags'. Arguments with '*' are required.

  --json-cfg JSON_CFG   * The path to the input JSON file with configuration of the inference

Inference configuration with flags:
  Configuration with flags. This section is not compatible with 'Inference configuration with JSON'. Arguments with '*' are required.

  --modelwrapper-cls MODELWRAPPER_CLS
                        * ModelWrapper-based class with inference implementation to import
  --dataset-cls DATASET_CLS
                        * Dataset-based class with dataset to import
  --compiler-cls COMPILER_CLS
                        * Optimizer-based class with compiling routines to import
  --runtime-cls RUNTIME_CLS
                        Runtime-based class with the implementation of model runtime
  --protocol-cls PROTOCOL_CLS
                        Protocol-based class with the implementation of communication between inference
                        tester and inference runner

ModelWrapper arguments:
  --model-path MODEL_PATH
                        Path to the model

PetDataset arguments:
  --classify-by {species,breeds}
                        Determines if classification should be performed by species or by breeds
  --image-memory-layout {NHWC,NCHW}
                        Determines if images should be delivered in NHWC or NCHW format

Dataset arguments:
  --dataset-root DATASET_ROOT
                        Path to the dataset directory
  --inference-batch-size INFERENCE_BATCH_SIZE
                        The batch size for providing the input data
  --download-dataset    Downloads the dataset before taking any action. If the dataset files are already downloaded and the checksum is correct then they are not downloaded again. Is enabled by default.
  --force-download-dataset
                        Forces dataset download
  --external-calibration-dataset EXTERNAL_CALIBRATION_DATASET
                        Path to the directory with the external calibration dataset
  --split-fraction-test SPLIT_FRACTION_TEST
                        Default fraction of data to leave for model testing
  --split-fraction-val SPLIT_FRACTION_VAL
                        Default fraction of data to leave for model valdiation
  --split-seed SPLIT_SEED
                        Default seed used for dataset split

TVMRuntime arguments:
  --save-model-path SAVE_MODEL_PATH
                        Path where the model will be uploaded
  --target-device-context {llvm,stackvm,cpu,c,test,hybrid,composite,cuda,nvptx,cl,opencl,sdaccel,aocl,aocl_sw_emu,vulkan,metal,vpi,rocm,ext_dev,hexagon,webgpu}
                        What accelerator should be used on target device
  --target-device-context-id TARGET_DEVICE_CONTEXT_ID
                        ID of the device to run the inference on
  --runtime-use-vm      At runtime use the TVM Relay VirtualMachine

Runtime arguments:
  --disable-performance-measurements
                        Disable collection and processing of performance metrics

TVMCompiler arguments:
  --model-framework {keras,onnx,darknet,torch,tflite}
                        The input type of the model, framework-wise
  --target TARGET       The kind or tag of the target device
  --target-host TARGET_HOST
                        The kind or tag of the host (CPU) target device
  --opt-level OPT_LEVEL
                        The optimization level of the compilation
  --libdarknet-path LIBDARKNET_PATH
                        Path to the libdarknet.so library, for darknet models
  --compile-use-vm      At compilation stage use the TVM Relay VirtualMachine
  --output-conversion-function {default,dict_to_tuple}
                        The type of output conversion function used for PyTorch conversion
  --conv2d-data-layout CONV2D_DATA_LAYOUT
                        Configures the I/O layout for the CONV2D operations
  --conv2d-kernel-layout CONV2D_KERNEL_LAYOUT
                        Configures the kernel layout for the CONV2D operations
  --use-fp16-precision  Applies conversion of FP32 weights to FP16
  --use-int8-precision  Applies conversion of FP32 weights to INT8
  --use-tensorrt        For CUDA targets: delegates supported operations to TensorRT
  --dataset-percentage DATASET_PERCENTAGE
                        Tells how much data from the calibration dataset (training or external) will be used for calibration dataset

Optimizer arguments:
  --compiled-model-path COMPILED_MODEL_PATH
                        The path to the compiled model output

NetworkProtocol arguments:
  --host HOST           The address to the target device
  --port PORT           The port for the target device

BytesBasedProtocol arguments:
  --packet-size PACKET_SIZE
                        The maximum size of the received packets, in bytes.
  --endianness {big,little}
                        The endianness of data to transfer
```

The `kenning.scenarios.inference_server` requires only:

* a [](protocol-api)-based class for the implementation of the communication,
* a [](runtime-api)-based class for the implementation of runtime routines on device.

Both classes may require some additional arguments that can be listed with the `-h` flag.

An example script for the `inference_tester` is:

```bash test-skip
kenning optimize test \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/google-coral-devboard-tflite-tensorflow.json \
    --compiler-cls kenning.optimizers.tflite.TFLiteCompiler \
    --protocol-cls kenning.protocols.network.NetworkProtocol \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "edgetpu" \
    --compiled-model-path build/compiled-model.tflite \
    --inference-input-type int8 \
    --inference-output-type int8 \
    --host 192.168.188.35 \
    --port 12344 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --dataset-root build/pet-dataset \
    --inference-batch-size 1 \
    --verbosity INFO
```

The above runs with the following `inference_server` setup:

```bash test-skip
kenning server \
    --protocol-cls kenning.protocols.network.NetworkProtocol \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --host 0.0.0.0 \
    --port 12344 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --delegates-list libedgetpu.so.1 \
    --verbosity INFO
```

```{note}
This run was tested on a Google Coral Devboard device.
```

`kenning.scenarios.inference_tester` can be also executed locally - in this case, the `--protocol-cls` argument can be skipped.
The example call is as follows:

```bash timeout=5
kenning optimize test \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/local-cpu-tvm-tensorflow-classification.json \
    --compiler-cls kenning.optimizers.tvm.TVMCompiler \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "llvm" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cpu \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
```

```{note}
For more examples of running `inference_tester` and `inference_server`, check the [kenning/scripts](https://github.com/antmicro/kenning/tree/main/scripts) directory.
Directories with scripts for client and server calls for various target devices, deep learning frameworks and compilation frameworks can be found in the [kenning/scripts/edge-runtimes](https://github.com/antmicro/kenning/tree/main/scripts/edge-runtimes) directory.
```

## Running inference

`kenning.scenarios.inference_runner` is used to run inference locally on a pre-compiled model.

`kenning.scenarios.inference_runner` requires:

* a [](modelwrapper-api)-based class that performs I/O processing specific to the model,
* a [](runtime-api)-based class that runs inference on target using the compiled model,
* a [](dataprovider-api)-based class that implements fetching of data samples from various sources,
* a list of [](outputcollector-api)-based classes that implement output processing for the specific use case.

To print the list of required arguments, run:

```bash test-skip
python3 -m kenning.scenarios.inference_runner \
    kenning.modelwrappers.object_detection.darknet_coco.TVMDarknetCOCOYOLOV3 \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.dataproviders.camera_dataprovider.CameraDataProvider \
    --output-collectors kenning.outputcollectors.name_printer.NamePrinter \
    -h
```

With the above classes, the help can look as follows:

```bash test-skip
positional arguments:
  modelwrappercls       ModelWrapper-based class with inference implementation to import
  runtimecls            Runtime-based class with the implementation of model runtime
  dataprovidercls       DataProvider-based class used for providing data
optional arguments:
  -h, --help            show this help message and exit
  --output-collectors OUTPUT_COLLECTORS [OUTPUT_COLLECTORS ...]
                        List to the OutputCollector-based classes where the results will be passed
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Verbosity level
Inference model arguments:
  --model-path MODEL_PATH
                        Path to the model
  --classes CLASSES     File containing Open Images class IDs and class names in CSV format to use (can be generated using
                        kenning.scenarios.open_images_classes_extractor) or class type
Runtime arguments:
  --disable-performance-measurements
                        Disable collection and processing of performance metrics
  --save-model-path SAVE_MODEL_PATH
                        Path where the model will be uploaded
  --target-device-context {llvm,stackvm,cpu,c,cuda,nvptx,cl,opencl,aocl,aocl_sw_emu,sdaccel,vulkan,metal,vpi,rocm,ext_dev,hexagon,webgpu}
                        What accelerator should be used on target device
  --target-device-context-id TARGET_DEVICE_CONTEXT_ID
                        ID of the device to run the inference on
  --input-dtype INPUT_DTYPE
                        Type of input tensor elements
  --runtime-use-vm      At runtime use the TVM Relay VirtualMachine
  --use-json-at-output  Encode outputs of models into a JSON file with base64-encoded arrays
DataProvider arguments:
  --video-file-path VIDEO_FILE_PATH
                        Video file path (for cameras, use /dev/videoX where X is the device ID eg. /dev/video0)
  --image-memory-layout {NHWC,NCHW}
                        Determines if images should be delivered in NHWC or NCHW format
  --image-width IMAGE_WIDTH
                        Determines the width of the image for the model
  --image-height IMAGE_HEIGHT
                        Determines the height of the image for the model
OutputCollector arguments:
  --print-type {detector,classificator}
                        What is the type of model that will input data to the NamePrinter
```

An example script for `inference_runner`:

```bash test-skip
python3 -m kenning.scenarios.inference_runner \
    kenning.modelwrappers.object_detection.darknet_coco.TVMDarknetCOCOYOLOV3 \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.dataproviders.camera_dataprovider.CameraDataProvider \
    --output-collectors kenning.outputcollectors.detection_visualizer.DetectionVisualizer kenning.outputcollectors.name_printer.NamePrinter \
    --disable-performance-measurements \
    --model-path kenning:///models/object_detection/yolov3.weights \
    --save-model-path ../compiled-model.tar \
    --target-device-context "cuda" \
    --verbosity INFO \
    --video-file-path /dev/video0
```

(report-generation)=

## Generating performance reports

`kenning.scenarios.inference_performance` and `kenning.scenarios.inference_tester` return a JSON file as the result of benchmarks.
They contain both performance metrics data, and quality metrics data.

The data from JSON files can be analyzed, processed and visualized by the `kenning.scenarios.render_report` script.
This script parses the information in JSON files and returns an RST file with the report, along with visualizations.

It requires:

* a JSON file with benchmark data,
* a report name for use in the RST file and for creating Sphinx refs to figures,
* an RST output file name,
* `--root-dir` specifying the root directory of the Sphinx documentation where the RST file will be embedded (it is used to compute relative paths),
* `--img-dir` specifying the path where the figures should be saved,
* `--report-types`, which is a list describing the types the report falls into.

An example call and the resulting RST file can be observed in [](sample-report).

As for now, the available report types are:

* `performance` - this is the most common report type that renders information about overall inference performance metrics, such as inference time, CPU usage, RAM usage, or GPU utilization,
* `classification` - this report is specific to the classification task, it renders classification-specific quality figures and metrics, such as confusion matrices, accuracy, precision, G-mean,
* `detection` - this report is specific to the detection task, it renders detection-specific quality figures and metrics, such as recall-precision curves or mean average precision.

## Displaying information about available classes

`kenning.scenarios.list_classes` and `kenning.scenarios.class_info` provide useful information about classes and can help in creating JSON scenarios.

`kenning.scenarios.list_classes` will list all available classes by default, though the output can be limited by providing positional arguments representing groups of modules: `optimizers`, `runners`, `dataproviders`, `datasets`, `modelwrappers`, `onnxconversions`, `outputcollectors`, `runtimes`.
The amount of information displayed can be controlled using `-v` and `-vv` flags.

To print available arguments run `python -m kenning.scenarios.list_classes -h`.

`kenning.scenarios.class_info` provides information about a class given in an argument. More precisely, it will display:

* module and class docstrings,
* dependencies along with the information whether they are available in the current python environment,
* supported input and output formats,
* arguments structure used in JSON configurations.

The script uses a module-like path to the file (e.g. `kenning.runtimes.tflite`), but optionally a class can be specified by adding it to the path like so: `kenning.runtimes.tflite.TFLiteRuntime`.
To get more detailed information, an optional `--load-class-with-args` argument can be passed. This needs all required class arguments to be provided, and that all dependencies are available.

For more detail, check `python -m kenning.scenarios.class_info -h`.
