import pytest
from typing import Final, Dict
from contextlib import nullcontext as does_not_raise
import numpy as np
import time
from copy import deepcopy

from kenning.core.flow import KenningFlow
from kenning.dataproviders.camera_dataprovider import CameraDataProvider
from kenning.outputcollectors.real_time_visualizers import BaseRealTimeVisualizer # noqa: 501
from kenning.runners.modelruntime_runner import ModelRuntimeRunner
from kenning.interfaces.io_interface import IOCompatibilityError

FLOW_STEPS: Final = 4

CAMERA_DATA_PROVIDER_NCHW_JSON = {
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
CAMERA_DATA_PROVIDER_NHWC_JSON = deepcopy(CAMERA_DATA_PROVIDER_NCHW_JSON)
CAMERA_DATA_PROVIDER_NHWC_JSON['parameters']['input_memory_layout'] = 'NHWC'

MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON = {
    "type": "kenning.runners.modelruntime_runner.ModelRuntimeRunner",
    "parameters": {
        "model_wrapper": {
            "type": "kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4",
            "parameters": {
                "model_path": "./kenning/resources/models/detection/yolov4.onnx"    # noqa: 501
            }
        },
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters":
            {
                "save_model_path": "./kenning/resources/models/detection/yolov4.onnx",  # noqa: 501
                "execution_providers": ["CPUExecutionProvider"]
            }
        }
    },
    "inputs": {
        "input": "cam_frame"
    },
    "outputs": {
        "detection_output": "predictions"
    }
}
MODEL_RUNTIME_RUNNER_ONNXYOLOV4_2_JSON = deepcopy(
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON
)
MODEL_RUNTIME_RUNNER_ONNXYOLOV4_2_JSON['outputs']['detection_output'] = 'predictions_2'  # noqa: 501
MODEL_RUNTIME_RUNNER_ONNXYOLOV4_REDEFINED_VARIABLE_JSON = deepcopy(
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON
)
MODEL_RUNTIME_RUNNER_ONNXYOLOV4_REDEFINED_VARIABLE_JSON['outputs']['detection_output'] = 'cam_frame'    # noqa: 501
MODEL_RUNTIME_RUNNER_ONNXYOLOV4_UNDEFINED_VARIABLE_JSON = deepcopy(
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON
)
MODEL_RUNTIME_RUNNER_ONNXYOLOV4_UNDEFINED_VARIABLE_JSON['inputs']['input'] = 'undefined_cam_frame'  # noqa: 501

MODEL_RUNTIME_RUNNER_ONNXYOLACT_JSON = {
    "type": "kenning.runners.modelruntime_runner.ModelRuntimeRunner",
    "parameters": {
        "dataset":
        {
            "type": "kenning.datasets.coco_dataset.COCODataset2017",
            "parameters":
            {
                "dataset_root": "./build/coco-dataset",
                "download_dataset": True
            }
        },
        "model_wrapper": {
            "type": "kenning.modelwrappers.instance_segmentation.yolact.YOLACT",    # noqa: 501
            "parameters": {
                "model_path": "./kenning/resources/models/instance_segmentation/yolact.onnx"    # noqa: 501
            }
        },
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters":
            {
                "save_model_path": "./kenning/resources/models/instance_segmentation/yolact.onnx",  # noqa: 501
                "execution_providers": ["CPUExecutionProvider"]
            }
        }
    },
    "inputs": {
        "input": "cam_frame"
    },
    "outputs": {
        "segmentation_output": "predictions"
    }
}
DETECTION_VISUALIZER_JSON = {
    "type": "kenning.outputcollectors.detection_visualizer.DetectionVisualizer",    # noqa: 501
    "parameters": {
        "output_width": 608,
        "output_height": 608,
        "save_to_file": True,
        "save_path": "out_1.mp4"
    },
    "inputs": {
        "frame": "cam_frame",
        "detection_input": "predictions"
    }
}
DETECTION_VISUALIZER_2_JSON = deepcopy(DETECTION_VISUALIZER_JSON)
DETECTION_VISUALIZER_2_JSON['inputs']['detection_input'] = 'predictions'
DETECTION_VISUALIZER_2_JSON['parameters']['save_path'] = 'out_2.mp4'
DETECTION_VISUALIZER_3_JSON = deepcopy(DETECTION_VISUALIZER_JSON)
DETECTION_VISUALIZER_3_JSON['inputs']['detection_input'] = 'predictions_2'
DETECTION_VISUALIZER_3_JSON['parameters']['save_path'] = 'out_3.mp4'

RT_DETECTION_VISUALIZER_JSON = {
    "type": "kenning.outputcollectors.real_time_visualizers.RealTimeDetectionVisualizer",   # noqa: 501
    "parameters": {
        "viewer_width": 512,
        "viewer_height": 512,
        "input_memory_layout": "NCHW",
        "input_color_format": "BGR"
    },
    "inputs": {
        "frame": "cam_frame",
        "input": "predictions"
    }
}
RT_SEGMENTATION_VISUALIZER_JSON = {
    "type": "kenning.outputcollectors.real_time_visualizers.RealTimeSegmentationVisualization", # noqa: 501
    "parameters": {
        "viewer_width": 512,
        "viewer_height": 512,
        "score_threshold": 0.4,
        "input_memory_layout": "NCHW",
        "input_color_format": "BGR"
    },
    "inputs": {
        "frame": "cam_frame",
        "input": "predictions"
    }
}

FLOW_SCENARIO_DETECTION = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON,
    DETECTION_VISUALIZER_JSON
]
FLOW_SCENARIO_RT_DETECTION = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON,
    RT_DETECTION_VISUALIZER_JSON
]
FLOW_SCENARIO_RT_SEGMENTATION = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON,
    RT_SEGMENTATION_VISUALIZER_JSON
]
FLOW_SCENARIO_COMPLEX = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_2_JSON,
    DETECTION_VISUALIZER_JSON,
    DETECTION_VISUALIZER_2_JSON,
    DETECTION_VISUALIZER_3_JSON
]
FLOW_SCENARIO_VALID = FLOW_SCENARIO_DETECTION
FLOW_SCENARIO_REDEFINED_VARIBLE = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_REDEFINED_VARIABLE_JSON,
    DETECTION_VISUALIZER_JSON
]
FLOW_SCENARIO_UNDEFINED_VARIBLE = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_UNDEFINED_VARIABLE_JSON,
    DETECTION_VISUALIZER_JSON
]
FLOW_SCENARIO_INCOMPATIBLE_IO = [
    CAMERA_DATA_PROVIDER_NHWC_JSON,
    MODEL_RUNTIME_RUNNER_ONNXYOLOV4_JSON,
    DETECTION_VISUALIZER_JSON
]


@pytest.fixture(autouse=True)
def mock_camera_fetch_input():
    """
    Mocks camera input - instead of camera frame returns random noise.
    """
    def fetch_input(self):
        return np.random.randint(
            low=0,
            high=255,
            size=(256, 256, 3),
            dtype=np.uint8
        )

    CameraDataProvider.fetch_input = fetch_input
    CameraDataProvider.prepare = lambda self: None
    CameraDataProvider.detach_from_source = lambda self: None


@pytest.fixture
def set_should_close_after_3_calls():
    """
    Mocks should_close method so that after 3 calls it returns True.
    """
    def should_close(self):
        should_close.calls += 1
        return should_close.calls >= 3

    should_close.calls = 0

    ModelRuntimeRunner.should_close = should_close


@pytest.fixture
def mock_dear_py_gui():
    """
    Mocks DearPyGui so that there is no GUI being showed.
    """
    def _gui_thread(self):
        while not self.stop:
            if self.thread_data:
                _ = self.thread_data.pop(0)
            time.sleep(.01)

    BaseRealTimeVisualizer._gui_thread = _gui_thread
    BaseRealTimeVisualizer.should_close = lambda self: False


class TestKenningFlowScenarios:

    @pytest.mark.parametrize(
        'json_scenario,expectation',
        [
            (FLOW_SCENARIO_VALID,
             does_not_raise()),
            (FLOW_SCENARIO_REDEFINED_VARIBLE,
             pytest.raises(Exception)),
            (FLOW_SCENARIO_UNDEFINED_VARIBLE,
             pytest.raises(Exception)),
            (FLOW_SCENARIO_INCOMPATIBLE_IO,
             pytest.raises(IOCompatibilityError)),
        ],
        ids=[
            'valid_scenario',
            'redefined_variable',
            'undefined_variable',
            'incompatible_IO'
        ])
    def test_load_kenning_flows(self, json_scenario: Dict, expectation):
        """
        Tests KenningFlow loading from JSON and runner's IO validation.
        """
        with expectation:
            flow = KenningFlow.from_json(json_scenario)

            flow.cleanup()

    @pytest.mark.usefixtures(
        'mock_dear_py_gui'
    )
    @pytest.mark.parametrize('json_scenario', [
            FLOW_SCENARIO_RT_DETECTION,
            FLOW_SCENARIO_RT_SEGMENTATION,
            FLOW_SCENARIO_DETECTION,
            FLOW_SCENARIO_COMPLEX,
        ],
        ids=[
            'realtime_detection_scenario',
            'realtime_segmentation_scenario',
            'detection_scenario',
            'complex_scenario'
        ])
    def test_run_kenning_flows(self, json_scenario: Dict):
        """
        Tests execution of example flows.
        """
        flow = KenningFlow.from_json(json_scenario)

        try:
            for _ in range(FLOW_STEPS):
                flow.init_state()
                flow.run_single_step()
        except Exception as e:
            pytest.fail(f'Error during flow run: {e}')
        finally:
            flow.cleanup()

    @pytest.mark.usefixtures(
        'set_should_close_after_3_calls'
    )
    def test_kenning_flow_close_when_runner_should_close(self):
        """
        Tests closing flow when some runner got exit indicator.
        """
        flow = KenningFlow.from_json(FLOW_SCENARIO_VALID)
        flow.run()
