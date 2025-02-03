# Kenning API

## Deployment API overview

```{figure} img/class-flow.png
---
name: class-flow
alt: Kenning core classes and interactions between them
align: center
---

Kenning core classes and interactions between them.
The green blocks represent the flow of input data passed to the model for inference.
The orange blocks represent the flow of model deployment, from training to inference on target device.
The grey blocks represent the inference results and metrics flow.
```

{{projecturl}} provides:

* a [](dataset-api) class - performs dataset download, preparation, input preprocessing, output postprocessing and model evaluation,
* a [](modelwrapper-api) class - trains the model, prepares the model, performs model-specific input preprocessing and output postprocessing, runs inference on host using a native framework,
* a [](optimizer-api) class - optimizes and compiles the model,
* a [](runtime-api) class - loads the model, performs inference on compiled model, runs target-specific processing of inputs and outputs, and runs performance benchmarks,
* a [](protocol-api) class - implements the communication protocol between the host and the target,
* a [](dataconverter-api) class - performs conversion of data between different formats used by surrounding blocks,
* a [](dataprovider-api) class - implements providing data for inference from such sources as camera, TCP connection, or others,
* a [](outputcollector-api) class - implements parsing and utilizing data coming from inference (such as displaying visualizations or sending results via TCP).

### Model processing

The orange blocks and arrows in {numref}`class-flow` represent a model's life cycle:

* the model is designed, trained, evaluated and improved - the training is implemented in the [](modelwrapper-api).
  ```{note}
  This is an optional step - an already trained model can also be wrapped and used.
  ```
* the model is passed to the [](optimizer-api) where it is optimized for given hardware and later compiled,
* during inference testing, the model is sent to the target using [](protocol-api),
* the model is loaded on target side and used for inference using [](runtime-api).

Once the development of the model is complete, the optimized and compiled model can be used directly on target device using [](runtime-api).

### I/O data flow

The data flow is represented in the {numref}`class-flow` with green blocks.
The input data flow is depicted using green arrows, and the output data flow is depicted using grey arrows.

Firstly, the input and output data is loaded from dataset files and processed.
Later, since every model has its specific input preprocessing and output postprocessing routines, the data is passed to the [](modelwrapper-api) methods in order to apply modifications.
During inference testing, the data is sent to and from the target using [](protocol-api).

Lastly, since [](runtime-api)s also have their specific representations of data, proper I/O processing is applied.

### Data flow reporting

Report rendering requires performance metrics and quality metrics.
The flow for this is presented with grey lines and blocks in {numref}`class-flow`.

On target side, performance metrics are computed and sent back to the host using the [](protocol-api), and later passed to report rendering.
After the output data goes through processing in the [](runtime-api) and [](modelwrapper-api), it is compared to the ground truth in the [](dataset-api) during model evaluation.
In the end, the results of model evaluation are passed to report rendering.

The final report is generated as an RST file containing figures, as can be observed in the [](./sample-report).

(kenningflow-api)=
## KenningFlow

`kenning.core.flow.KenningFlow` class allows for creation and execution of arbitrary flows built of runners.
It is responsible for validating all runners provided in a config file and their IO compatibility.

```{eval-rst}
.. autoclass:: kenning.core.flow.KenningFlow
   :show-inheritance:
   :members:
```

(runner-api)=
## Runner

`kenning.core.runner.Runner` - based classes are responsible for executing various operation in KenningFlow (i.e. data providing, model execution, data visualization).

The available runner implementations are:
* [DataProvider](dataprovider-api) - base class for data providing,
* [ModelRuntimeRunner](https://github.com/antmicro/kenning/blob/main/kenning/runners/modelruntime_runner.py) - for running model inference,
* [OutputCollector](outputcollector-api) - for processing model output.

```{eval-rst}
.. autoclass:: kenning.core.runner.Runner
   :show-inheritance:
   :members:
```

(dataset-api)=
## Dataset

`kennning.core.dataset.Dataset` - based classes are responsible for:

* dataset preparation, including download routines (use the `--download-dataset` flag to download the dataset data),
* input preprocessing into a format expected by most models for a given task,
* output postprocessing for the evaluation process,
* model evaluation based on its predictions,
* sample subdivision into training and validation datasets.

The Dataset objects are used by:

* [](modelwrapper-api) - for training purposes and model evaluation,
* [](optimizer-api) - can be used e.g. for extracting a calibration dataset for quantization purposes,
* [](runtime-api) - for model evaluation on target hardware.

The available dataset implementations are included in the `kenning.datasets` submodule.
Example implementations:

* [PetDataset](https://github.com/antmicro/kenning/blob/main/kenning/datasets/pet_dataset.py) for classification,
* [OpenImagesDatasetV6](https://github.com/antmicro/kenning/blob/main/kenning/datasets/open_images_dataset.py) for object detection,
* [RandomizedClassificationDataset](https://github.com/antmicro/kenning/blob/main/kenning/datasets/random_dataset.py).

```{eval-rst}
.. autoclass:: kenning.core.dataset.Dataset
   :show-inheritance:
   :members:
```

(modelwrapper-api)=
## ModelWrapper

`kenning.core.model.ModelWrapper` base class requires implementing methods for:

* model preparation,
* model saving and loading,
* model saving to the ONNX format,
* model-specific preprocessing of inputs and postprocessing of outputs, if necessary,
* model inference,
* providing metadata (framework name and version),
* model training,
* input format specification,
* conversion of model inputs and outputs to bytes for the `kenning.core.protocol.Protocol` objects.

The `ModelWrapper` provides methods for running inference in a loop for data from a dataset and measuring the model's quality and inference performance.

The `kenning.modelwrappers.frameworks` submodule contains framework-wise implementations of the `ModelWrapper` class - they implement all methods common for given frameworks regardless of the model used.

For the `Pet Dataset wrapper` object, there is an example classifier implemented in TensorFlow 2.x called `TensorFlowPetDatasetMobileNetV2 <https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/tensorflow_pet_dataset.py>`_.

Model wrapper examples:

* [PyTorchWrapper](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/frameworks/pytorch.py) and [TensorFlowWrapper](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/frameworks/tensorflow.py) implement common methods for all PyTorch and TensorFlow framework models,
* [PyTorchPetDatasetMobileNetV2](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/pytorch_pet_dataset.py) wraps the MobileNetV2 model for Pet classification implemented in PyTorch,
* [TensorFlowDatasetMobileNetV2](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/tensorflow_pet_dataset.py) wraps the MobileNetV2 model for Pet classification implemented in TensorFlow,
* [TVMDarknetCOCOYOLOV3](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/object_detection/darknet_coco.py) wraps the YOLOv3 model for COCO object detection implemented in Darknet (without training and inference methods).

```{eval-rst}
.. autoclass:: kenning.core.model.ModelWrapper
   :show-inheritance:
   :members:
```

(optimizer-api)=
## Optimizer

`kenning.core.optimizer.Optimizer` objects wrap the deep learning compilation process.
They can perform the model optimization (operation fusion, quantization) as well.
Kenning supports executing optimizations also on the target device.
To do so you can use `location` parameter which specifies where given [](optimizer-api) would be executed (either `'host'` or `'target'`).

All Optimizer objects should provide methods for compiling models in ONNX format, but they can also provide support for other formats (like Keras .h5 files, or PyTorch .th files).

Example model optimizers:

* [TFLiteCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tflite.py) - wraps TensorFlow Lite compilation,
* [TVMCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tvm.py) - wraps TVM compilation.

```{eval-rst}
.. autoclass:: kenning.core.optimizer.Optimizer
   :show-inheritance:
   :members:
```

(runtime-api)=
## Runtime

The `kenning.core.runtime.Runtime` class provides interfaces for methods for running compiled models locally or remotely on a target device.
Runtimes are usually compiler-specific (frameworks for deep learning compilers provide runtime libraries to run compiled models on particular hardware).

The client (host) side of the `Runtime` class utilizes the methods from [](dataset-api), [](modelwrapper-api) and [](protocol-api) classes to run inference on a target device.
The server (target) side of the `Runtime` class requires method implementation for:

* loading a model delivered by the client,
* preparing inputs delivered by the client,
* running inference,
* preparing outputs for delivery to the client,
* (optionally) sending inference statistics.

Runtime examples:

* [TFLiteRuntime](https://github.com/antmicro/kenning/blob/main/kenning/runtimes/tflite.py) for models compiled with TensorFlow Lite,
* [TVMRuntime](https://github.com/antmicro/kenning/blob/main/kenning/runtimes/tvm.py) for models compiled with TVM.

```{eval-rst}
.. autoclass:: kenning.core.runtime.Runtime
   :show-inheritance:
   :members:
```

(protocol-api)=
## Protocol

The `kenning.core.protocol.Protocol` class conducts communication between the client (host) and the server (target).

The Protocol class requires method implementation for:

* initializing the server and the client (communication-wise),
* waiting for the incoming data,
* data sending,
* data receiving,
* uploading model inputs to the server,
* uploading the model to the server,
* requesting inference on target,
* downloading outputs from the server,
* (optionally) downloading the statistics from the server (e.g. performance speed, CPU/GPU utilization, power consumption),
* success or failure notifications from the server,
* message parsing.

Based on the above-mentioned methods, the `kenning.core.runtime.Runtime` connects the host with the target.

Protocol examples:

* [NetworkProtocol](https://github.com/antmicro/kenning/blob/main/kenning/protocols/network.py) - implements a TCP-based communication between the host and the client.

(protocol-spec)=
### Protocol specification

The communication protocol is message-based.
Possible messages are:

* `OK` messages - indicate success, and may come with additional information,
* `ERROR` messages - indicate failure,
* `DATA` messages - provide input data for inference,
* `MODEL` messages - provide model to load for inference,
* `PROCESS` messages - request processing inputs delivered in `DATA` message,
* `OUTPUT` messages - request processing results,
* `STATS` messages - request statistics from the target device.

The message types and enclosed data are encoded in a format implemented in the `kenning.core.protocol.Protocol`-based class.

Communication during an inference benchmark session goes as follows:

* The client (host) connects to the server (target),
* The client sends a `MODEL` request along with the compiled model,
* The server loads the model from request, prepares everything to run the model and sends an `OK` response,
* After receiving the `OK` response from the server, the client starts reading input samples from the dataset, preprocesses the inputs, and sends a `DATA` request with the preprocessed input,
* Upon receiving the `DATA` request, the server stores the input for inference, and sends an `OK` message,
* Upon receiving confirmation, the client sends a `PROCESS` request,
* Just after receiving the `PROCESS` request, the server should send an `OK` message to confirm start of inference, and just after the inference is finished, the server should send another `OK` message to confirm that the inference has finished,
* After receiving the first `OK` message, the client starts measuring inference time until the second `OK` response is received,
* The client sends an `OUTPUT` request in order to receive the outputs from the server,
* The server sends an `OK` message along with the output data,
* The client parses the output and evaluates model performance,
* The client sends a `STATS` request to obtain additional statistics (inference time, CPU/GPU/Memory utilization) from the server,
* If the server provides any statistics, it sends an `OK` message with the data,
* The same process applies to the rest of input samples.

The way the message type is determined and the data between the server and the client is sent depends on the implementation of the `kenning.core.protocol.Protocol` class.
The implementation of running inference on the given target is contained within the `kenning.core.runtime.Runtime` class.

### Protocol

`kenning.core.protocol.Protocol`-based classes implement the [](protocol-spec) in a given means of transport, e.g. TCP connection or UART.
It requires method implementation for:

* server (target hardware) and client (compiling host) initialization,
* sending and receiving data,
* connecting and disconnecting,
* model upload (host) and download (target hardware),
* message parsing and creation.

```{eval-rst}
.. autoclass:: kenning.core.protocol.Protocol
   :show-inheritance:
   :members:
```

(dataconverter-api)=
## DataConverter

`kennning.core.dataconverter.DataConverter` - based classes are responsible for:

* converting data to the format expected by surrounding block,
* converting data from the surrounding block format to one previous block excepts.

The `DataConverter` objects are used by [PipelineRunner](https://github.com/antmicro/kenning/blob/main/kenning/utils/pipeline_runner.py) during inference.

The available implementations of dataconverter are included in the `kenning.dataconverters` submodule.
Example implementations:

* [ModelWrapperDataConverter](https://github.com/antmicro/kenning/blob/main/kenning/dataconverters/modelwrapper_dataconverter.py) implement conversions by utilizing the [ModelWrapper](modelwrapper-api) classes,
* [ROS2SegmentationDataConverter](https://github.com/antmicro/kenning/blob/main/kenning/dataconverters/ros2_segmentation_dataconverter.py) performs conversions by utilizing ROS2 action for segmentation problems from [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs) repository.

```{eval-rst}
.. autoclass:: kenning.core.dataconverter.DataConverter
   :show-inheritance:
   :members:
```

(measurements-api)=
## Measurements

The `kenning.core.measurements` module contains `Measurements` and `MeasurementsCollector` classes for collecting performance and quality metrics.
`Measurements` is a dict-like object that provides various methods for adding performance metrics, adding values for time series, and updating existing values.

The dictionary held by `Measurements` requires serializable data, since most scripts save performance results in JSON format for later report generation.

```{eval-rst}
.. automodule:: kenning.core.measurements
   :members:
```

(onnxconversion-api)=
## ONNXConversion

The `ONNXConversion` object contains methods for model conversion in various frameworks to ONNX and vice versa.
It also provides methods for testing the conversion process empirically on a list of deep learning models implemented in the tested frameworks.

```{eval-rst}
.. autoclass:: kenning.core.onnxconversion.ONNXConversion
   :show-inheritance:
   :members:
```

(dataprovider-api)=
## DataProvider

The `DataProvider` classes are used during deployment to provide data for inference.
They can provide data from such sources as a camera, video files, microphone data or a TCP connection.

The available `DataProvider` implementations are included in the `kenning.dataproviders` submodule.
Example implementations:

* [CameraDataProvider](https://github.com/antmicro/kenning/blob/main/kenning/dataproviders/camera_dataprovider.py) for capturing frames from camera.

```{eval-rst}
.. autoclass:: kenning.core.dataprovider.DataProvider
   :show-inheritance:
   :members:
```

(outputcollector-api)=
## OutputCollector

The `OutputCollector` classes are used during deployment for inference results receiving and processing.
They can display the results, send them, or store them in a file.

The available output collector implementations are included in the `kenning.outputcollectors` submodule.
Example implementations:

* [DetectionVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/detection_visualizer.py) for visualizing detection model outputs,
* [BaseRealTimeVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/real_time_visualizers.py) base class for real time visualizers:
   * [RealTimeDetectionVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/real_time_visualizers.py)
      for visualizing detection model outputs,
   * [RealTimeSegmentationVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/real_time_visualizers.py)
      for visualizing segmentation model outputs,
   * [RealTimeClassificationVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/real_time_visualizers.py)
      for visualizing classification model outputs.


```{eval-rst}
.. autoclass:: kenning.core.outputcollector.OutputCollector
   :show-inheritance:
   :members:
```

(argumentshandler-api)=
## ArgumentsHandler

The `ArgumentsHandler` class is responsible for concatenating `arguments_structure` and creating parsers for command line and JSON config arguments.

In order to make some class being able to be instantiated from command line arguments or JSON config it is required to inherit from this class or its child class and implement `from_argparse` or `from_json` methods as described in [](defining-arguments-for-core-classes).

```{eval-rst}
.. autoclass:: kenning.utils.args_manager.ArgumentsHandler
   :show-inheritance:
   :members:
```

(resourcemanager-api)=
## ResourceManager

The `ResourceManager` is a singleton class which handles local and remote files for Kenning, such as datasets or models.
It downloads missing files, provides paths to the available files, resolves custom URL schemes for and manages resource directories for Kenning, including cleanup.

```{eval-rst}
.. autoclass:: kenning.utils.resource_manager.ResourceManager
   :show-inheritance:
   :members:
```

(resourceuri-api)=
## ResourceURI

The `ResourceURI` class is a [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)-based object allowing the user to work with Kenning resources, both using regular paths and URI schemes supported by the [](resourcemanager-api).

```{eval-rst}
.. autoclass:: kenning.utils.resource_manager.ResourceURI
   :show-inheritance:
   :members:
```
