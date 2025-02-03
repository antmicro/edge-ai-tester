# Choosing optimal optimization pipeline

The {{optimization_runner_script}} script allows to optimize over multiple pipelines and to choose the best performing based on a specified criteria

The script can be run as follows:

```bash test-skip
kenning fine-tune-optimizers --json-cfg <CONFIG_JSON>.json --output <OUTPUT_PATH>.json
```

With the arguments:
* `<CONFIG_JSON>.json` - describes the configuration, which pipelines would be executed and optimization settings
* `<OUTPUT_PATH>.json` - base path to the output files with measurements.

## Optimization config specification

The example configuration looks like this:

{ emphasize-lines="2-8" }
```json
{
    "optimization_parameters":
    {
        "strategy": "grid_search",
        "optimizable": ["optimizers", "runtime"],
        "metric": "inferencetime_mean",
        "policy": "min"
    },
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
            "type": "kenning.optimizers.tflite.TFLiteCompiler",
            "parameters":
            {
                "target": ["default"],
                "compiled_model_path": ["./build/compiled_model.tflite"]
            }
        },
        {
            "type": "kenning.optimizers.tvm.TVMCompiler",
            "parameters":
            {
                "target": ["llvm"],
                "compiled_model_path": ["./build/compiled_model.tar"],
                "opt_level": [3],
                "conv2d_data_layout": ["NHWC", "NCHW"]
            }
        }
    ],
    "runtime":
    [
        {
            "type": "kenning.runtimes.tvm.TVMRuntime",
            "parameters":
            {
                "save_model_path": ["./build/compiled_model.tar"]
            }
        },
        {
            "type": "kenning.runtimes.tflite.TFLiteRuntime",
            "parameters":
            {
                "save_model_path": ["./build/compiled_model.tflite"]
            }
        }
    ]
}
```

The highlighted part describes the settings of the optimization run.
* `strategy` - Describes how the pipelines are chosen for optimization. Currently only available option is `grid_search`, which generates cartesian product of all blocks that should be optimized and runs all compatible pipelines
* `optimizable` - Which Kenning block should be variable in the pipeline. The {{optimization_runner_script}} script will choose the Kenning block out of multiple provided in the later part of configuration based on `strategy`.
* `metric` - one of the metrics from [Kenning measurements](kenning-measurements) over which the optimization should be performed
* `policy` - `min` or `max`, depending on whether the metric should be maximized or minimized

The rest of the JSON configuration describes the pipeline in a format similar to [standard JSON scenarios](json-scenarios). The only difference is within the blocks chosen in the `optimizable` field, which should be list of blocks instead of singular definition. Every block in the list has defined list of arguments for all of it's parameters, this allows to choose the best argument for a particular block option
* For `optimizers` block, since there can be more than one in the pipeline, all possible combinations are tested.
* For the rest of the blocks one of all possible blocks is chosen for each run.

Kenning checks automatically whether a selected combination of blocks (model, optimizations and runtime) is compatible.

## Output details

The {{optimization_runner_script}} outputs multiple JSON files:

* `<OUTPUT PATH>.json` - Most optimal pipeline with it's aggregated metrics
* `<OUTPUT PATH>_all_results.json` - All pipeline configs and their respective metrics gathered into single file
* `<OUTPUT PATH>_<RUN ID>.json` - Full benchmark output for every optimization run.

The example `<OUTPUT PATH>.json` JSON file can look like this:

```json
{
    "pipeline": {
        "model_wrapper": {
            "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
            "parameters": {
                "model_path": "kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
            }
        },
        "dataset": {
            "type": "kenning.datasets.pet_dataset.PetDataset",
            "parameters": {
                "dataset_root": "./build/pet-dataset"
            }
        },
        "optimizers": [
            {
                "type": "kenning.optimizers.tvm.TVMCompiler",
                "parameters": {
                    "target": "llvm",
                    "compiled_model_path": "build/7_compiled_model.tar",
                    "opt_level": 3,
                    "conv2d_data_layout": "NCHW"
                }
            }
        ],
        "runtime": {
            "type": "kenning.runtimes.tflite.TFLiteRuntime",
            "parameters": {
                "save_model_path": "build/7_compiled_model.tflite"
            }
        }
    },
    "metrics": {
        "inferencetime_mean": 0.0031105340278004203,
        "inferencetime_std": 0.0003073369739186006,
        "inferencetime_median": 0.003066142999159638,
        "session_utilization_mem_percent_mean": 26.089648437500003,
        "session_utilization_mem_percent_std": 0.2059426531921169,
        "session_utilization_mem_percent_median": 26.1,
        "session_utilization_cpus_percent_avg_mean": 16.55806884765625,
        "session_utilization_cpus_percent_avg_std": 4.068584532658253,
        "session_utilization_cpus_percent_avg_median": 15.85625,
        "accuracy": 0.957273098380732,
        "mean_precision": 0.9574413909275928,
        "mean_sensitivity": 0.9571990915922841,
        "g_mean": 0.9566448082813757,
        "top_5_accuracy": 0.9961899578173902
    }
}
```

* The `pipeline` contains the definition of the best optimization pipeline.
* The `metrics` contains list of all aggregated metrics that can be considered in `metric` field `optimization_parameters`.
