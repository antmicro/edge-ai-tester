# Creating applications with Kenning

The [](kenningflow-api) allows you to run an arbitrary sequence of processing blocks that provide data, execute models using existing Kenning classes and wrappers, and processes results.
You can use it to quickly create applications with Kenning after optimizing the model and using it in actual use cases.

Kenning for runtime uses both existing classes, such as [](modelwrapper-api), [](runtime-api), and dedicated [](runner-api)-based classes.
The latter family of classes are actual functional blocks used in [](kenningflow-api) that can be used for:

* Obtaining data from sources - [](dataprovider-api), e.g. iterating files in the filesystem, grabbing frames from a camera or downloading data from a remote source,
* Processing and delivering data - [](outputcollector-api), e.g. sending results to the client application, visualizing model results in GUI, or storing results in a summary file,
* Running and processing various models,
* Applying other actions, such as additional data analysis, preprocessing, packing, and more.

A [](kenningflow-api) scenario definition can be saved in a JSON file and then run using the {{json_flow_runner_script}} script.

## JSON structure

JSON configuration consist of a list of dictionaries describing each [](runner-api)-based instance.

A sample [](runner-api) specification looks as follows:

```json
{
  "type": "kenning.dataproviders.camera_dataprovider.CameraDataProvider",
  "parameters": {
    "video_file_path": "/dev/video0",
    "input_memory_layout": "NCHW",
    "input_width": 608,
    "input_height": 608
  },
  "outputs": {
    "frame": "cam_frame"
  }
}
```

Each [](runner-api) dictionary consists of:

* `type` - [](runner-api) class. E.g. [CameraDataProvider](https://github.com/antmicro/kenning/blob/main/kenning/dataproviders/camera_dataprovider.py),
* `parameters` - parameters passed to class constructor. In this case, we specify a path to a video device (`/dev/video0`), expected memory format (`NCHW`), image size (`608x608`)
* `inputs` - (*optional*) [](runner-api) instance inputs. In the example above, there are none,
* `outputs` - (*optional*) [](runner-api) instance outputs. In the example above, it is a single output - camera frame defined as the variable `cam_frame` in the flow.

### Runner IO

The input and output specification in [](runner-api) classes is the same as described in [](model-io-metadata).

### IO compatibility

IO compatibility is checked during flow JSON parsing.

The [](runner-api) input is considered to be compatible with associated outputs if:

* in case of `numpy.ndarray`: `dtype` and `ndim` are equal and each dimension has either the same length or input dimension is set as `-1`, which represents any length.
  In the input spec, there can also be multiple valid shapes. If so, they are placed in an array, i.e. `[(1, -1, -1, 3), (1, 3, -1, -1)]`,
* in other cases: `type` fields are either equal or the input `type` field is `Any`.

### IO non-standard types

If the input or output is not a `numpy.ndarray`, then its type is described by the `type` field, which is a string.
In the case of a detection output from an IO specification (described above) it is a `List[DetectObject]`.
This is interpreted as a list of `DetectObject`s.
The `DetectObject` is a named tuple describing detection output (class names, rectangle positions, score).

### IO names and mapping

The inputs and outputs present in JSON are mappings from [](runner-api)'s local IO names to flow global variable names, i.e. one [](runner-api) can define its outputs as `{"output_name": "data"}` and another runner can use it as its input with `{"input_name": "data"}`.
These global variables must be unique and the variable defined as input needs to be defined in a previous block as output to prevent cycles in a flow's structure.
[](runner-api) IO names are specific to runner type and model (for `ModelRuntimeRunner`).

```{note}
IO names can be obtained using the `get_io_specification` method.
```

### Runtime example

In order to create a [](kenningflow-api) presenting YOLOv4 model performance, create a file `flow_scenario_detection.json` and include the following configuration in it:

```json
[
  {
    "type": "kenning.dataproviders.camera_dataprovider.CameraDataProvider",
    "parameters": {
      "video_file_path": "/dev/video0",
      "input_memory_layout": "NCHW",
      "input_width": 608,
      "input_height": 608
    },
    "outputs": {
      "frame": "cam_frame"
    }
  },
  {
    "type": "kenning.runners.modelruntime_runner.ModelRuntimeRunner",
    "parameters": {
      "model_wrapper": {
        "type": "kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4",
        "parameters": {
          "model_path": "kenning:///models/detection/yolov4.onnx"
        }
      },
      "runtime": {
        "type": "kenning.runtimes.onnx.ONNXRuntime",
        "parameters":
        {
          "save_model_path": "kenning:///models/detection/yolov4.onnx",
          "execution_providers": ["CUDAExecutionProvider"]
        }
      }
    },
    "inputs": {
      "input": "cam_frame"
    },
    "outputs": {
      "detection_output": "predictions"
    }
  },
  {
    "type": "kenning.outputcollectors.real_time_visualizers.RealTimeDetectionVisualizer",
    "parameters": {
      "viewer_width": 512,
      "viewer_height": 512,
      "input_memory_layout": "NCHW",
      "input_color_format": "BGR"
    },
    "inputs": {
      "frame": "cam_frame",
      "detection_data": "predictions"
    }
  }
]
```

This JSON creates a [](kenningflow-api) that consists of three runners - [CameraDataProvider](https://github.com/antmicro/kenning/blob/main/kenning/dataproviders/camera_dataprovider.py), [ModelRuntimeRunner](https://github.com/antmicro/kenning/blob/main/kenning/runners/modelruntime_runner.py) and [RealTimeDetectionVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/real_time_visualizers.py):

* The first one captures frames from a camera and passes it as a `cam_frame` variable.
* The next one passes `cam_frame` to a detection model (in this case YOLOv4) and returns predicted detection objects as `predictions`.
* The last one gets both outputs (`cam_frame` and `predictions`) and shows a detection visualization using DearPyGui.

## KenningFlow execution

Now, you can execute KenningFlow using the above configuration.

With the config saved in the `flow_scenario_detection.json` file, run the {{json_flow_runner_script}} as follows:

```bash test-skip
kenning flow --json-cfg flow_scenario_detection.json
```

This module runs KenningFlow defined in given JSON file.
With provided config it should read image from the camera and visualize output of detection model YOLOv4.

## Implemented Runners

Available implementations of [](runner-api) can be found in the [Runner documentation](runner-api).

To create custom runners, check [](implementing-runner).
