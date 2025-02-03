
## Pet Dataset classification using TVM-compiled TensorFlow model


### Commands used

````{note}

This section was generated using:

```bash
python -m kenning.__main__ \
    optimize \
    test \
    --modelwrapper-cls \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls \
        kenning.datasets.pet_dataset.PetDataset \
    --measurements \
        ./build/local-cpu-tvm-tensorflow-classification.json \
    --compiler-cls \
        kenning.optimizers.tvm.TVMCompiler \
    --runtime-cls \
        kenning.runtimes.tvm.TVMRuntime \
    --model-path \
        kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework \
        keras \
    --target \
        llvm \
    --compiled-model-path \
        ./build/compiled-model.tar \
    --opt-level \
        3 \
    --save-model-path \
        ./build/compiled-model.tar \
    --target-device-context \
        cpu \
    --dataset-root \
        ./build/PetDataset/ \
    --inference-batch-size \
        1 \
    --verbosity \
        INFO

python -m kenning.__main__ \
    report \
    --report-path \
        docs/source/generated/local-cpu-tvm-tensorflow-classification.md \
    --report-name \
        Pet Dataset classification using TVM-compiled TensorFlow model \
    --root-dir \
        docs/source/ \
    --img-dir \
        docs/source/generated/img \
    --report-types \
        performance \
        classification \
    --measurements \
        build/local-cpu-tvm-tensorflow-classification.json \
    --smaller-header

```
````

### General information for build.local-cpu-tvm-tensorflow-classification.json

*Model framework*:

* tensorflow ver. 2.11.1

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters": {
            "classify_by": "breeds",
            "image_memory_layout": "NHWC",
            "dataset_root": "build/PetDataset",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.2,
            "split_fraction_val": null,
            "split_seed": 1234
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters": {
            "model_path": "/home/runner/.kenning/models/classification/tensorflow_pet_dataset_mobilenetv2.h5",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tvm.TVMRuntime",
        "parameters": {
            "save_model_path": "build/compiled-model.tar",
            "target_device_context": "cpu",
            "target_device_context_id": 0,
            "runtime_use_vm": false,
            "llext_binary_path": null,
            "disable_performance_measurements": false
        }
    },
    "data_converter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tvm.TVMCompiler",
            "parameters": {
                "model_framework": "keras",
                "target": "llvm",
                "target_attrs": "",
                "target_microtvm_board": null,
                "target_host": null,
                "zephyr_header_template": null,
                "zephyr_llext_source_template": null,
                "opt_level": 3,
                "libdarknet_path": "/usr/local/lib/libdarknet.so",
                "compile_use_vm": false,
                "output_conversion_function": "default",
                "conv2d_data_layout": "",
                "conv2d_kernel_layout": "",
                "use_fp16_precision": false,
                "use_int8_precision": false,
                "use_tensorrt": false,
                "dataset_percentage": 0.25,
                "compiled_model_path": "build/compiled-model.tar",
                "location": "host"
            }
        }
    ]
}

```
## Inference performance metrics for build.local-cpu-tvm-tensorflow-classification.json


### Inference time

```{figure} generated/img/inference_time.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_inferencetime
alt: Inference time
align: center
---

Inference time
```
* *First inference duration* (usually including allocation time): **0.0772869379999861**,
* *Mean*: **0.07418959635646176 s**,
* *Standard deviation*: **0.0004082952863268992 s**,
* *Median*: **0.07411699150000572 s**.


### Average CPU usage

```{figure} generated/img/cpu_usage.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_cpuusage
alt: Average CPU usage
align: center
---

Average CPU usage during benchmark
```

* *Mean*: **50.6328637510513 %**,
* *Standard deviation*: **2.474337415693573 %**,
* *Median*: **50.0 %**.


### Memory usage

```{figure} generated/img/cpu_memory_usage.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_memusage
alt: Memory usage
align: center
---

Memory usage during benchmark
```

* *Mean*: **9.064507989907485 %**,
* *Standard deviation*: **0.047848910424839235 %**,
* *Median*: **9.1 %**.





## Inference quality metrics for build.local-cpu-tvm-tensorflow-classification.json


```{figure} generated/img/confusion_matrix.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_confusionmatrix
alt: Confusion matrix
align: center
---

Confusion matrix
```

* *Accuracy*: **0.9578231292517007**
* *Top-5 accuracy*: **0.9959183673469387**
* *Mean precision*: **0.9591048246905213**
* *Mean sensitivity*: **0.9572500515056033**
* *G-mean*: **0.9563343851604948**
