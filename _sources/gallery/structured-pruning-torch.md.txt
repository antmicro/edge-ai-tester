# Structured pruning for PyTorch models

Structured pruning is of the methods for reducing model size, which removes the least contributing neurons, filters and/or convolution kernels.
In this example, we present scenarios for structured pruning of [`PyTorchPetDatasetMobileNetV2`](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/pytorch_pet_dataset.py) using [Neural Network Intelligence](https://github.com/microsoft/nni).

## Setup

Install required dependencies:

```bash
pip install "kenning[nni,reports] @ git+https://github.com/antmicro/kenning.git"
```

## Experiments

In order to compare with an original model, you need to execute a scenario:

```{literalinclude} ../scripts/jsonconfigs/mobilenetv2-pytorch.json save-as=mobilenetv2-pytorch.json
:language: json
```

To run it, use this command:

```bash
kenning test \
  --json-cfg mobilenetv2-pytorch.json \
  --measurements build/torch.json
```

{{project}} supports activation-based pruners.
You can choose a specific pruner with the `pruner_type` parameter:

* `apoz` - [`ActivationAPoZRankPruner`](https://nni.readthedocs.io/en/v2.5/Compression/v2_pruning_algo.html#activation-apoz-rank-pruner) based on Average Percentage of Zeros in activations,
* `mean_rank` - [`ActivationMeanRankPruner`](https://nni.readthedocs.io/en/v2.5/Compression/v2_pruning_algo.html#activation-mean-rank-pruner) based on a metric that calculates the smallest mean value of activations.

These activations are collected during dataset inference.
The number of samples collected for statistics can be modified with `training_steps`.
Moreover, pruning has two modes:

* `dependency_aware` - makes pruner aware of dependencies for channels and groups.
* `normal` - dependencies are ignored.

You can also choose which `activation` the pruner will use - `relu`, `relu6` or `gelu`.
Additional configuration can be specified in `config_list` which follows a format defined in the [NNI specification](https://nni.readthedocs.io/en/stable/compression/config_list.html#pruning-specific-configuration-keys).
When `exclude_last_layer` is positive, the optimizer will be configured to exclude the last layer from the pruning process, to prevent changing the size of the output.
Apart from that, `confidence` defines the coefficient for sparsity inference and batch size of the dummy input for the process.
When a GPU is available, it is used by default, but as pruning can be memory-consuming, the `pruning_on_cuda` option enables manual GPU usage configuration during the process.

Other arguments affect fine-tuning of the pruned model, e.g. `criterion` and `optimizer` accept paths to classes, respectively calculating a criterion and optimizing a neural network.
The number of `finetuning_epochs`, the `finetuning_batch_size` and `finetuning_learning_rate` can be modified.

```{literalinclude} ../scripts/jsonconfigs/pruning-mobilenetv2-pytorch.json save-as=pruning-mobilenetv2-pytorch.json
:language: json
:emphasize-lines: 16-44
```

Run the above scenario with:

```bash
kenning optimize test \
  --json-cfg pruning-mobilenetv2-pytorch.json \
  --measurements build/nni-pruning.json
```

To ensure better quality of performance measurements, we suggest running optimization and tests separately, like below:

```bash test-skip
kenning optimize --json-cfg pruning-mobilenetv2-pytorch.json
kenning test \
  --json-cfg pruning-mobilenetv2-pytorch.json \
  --measurements build/nni-pruning.json
```

For more size reduction, you can use larger sparsity with adjusted parameters, like below:

{ emphasize-lines="7,14,18-19,24" }
```json
{
  "type": "kenning.optimizers.nni_pruning.NNIPruningOptimizer",
  "parameters": {
    "pruner_type": "mean_rank",
    "config_list": [
      {
        "total_sparsity": 0.15,
        "op_types": [
          "Conv2d",
          "Linear"
        ]
      }
    ],
    "training_steps": 92,
    "activation": "relu",
    "compiled_model_path": "build/nni-pruning-0_15.pth",
    "mode": "dependency_aware",
    "finetuning_epochs": 10,
    "finetuning_batch_size": 64,
    "confidence": 8,
    "criterion": "torch.nn.CrossEntropyLoss",
    "optimizer": "torch.optim.Adam",
    "pruning_on_cuda": true,
    "finetuning_learning_rate": 1e-04,
    "exclude_last_layer": true
  }
}
```

## Results

Models can be compared with the generated report:

```bash
kenning report \
  --measurements \
    build/torch.json \
    build/nni-pruning.json \
  --report-path build/nni-pruning.md \
  --to-html
```

Pruned models have significantly fewer parameters, which results in decreased GPU and VRAM usage without increasing inference time.
Summary of a few examples with different sparsity:

| Sparsity      | Accuracy     | Number of parameters | Fine-tuning epochs | Size reduction |
|---------------|--------------|----------------------|--------------------|----------------|
| ---           | 0.9632653061 |            4,130,853 |                --- |          0.00% |
| 0.05          | 0.9299319728 |            3,660,421 |                  3 |         11.27% |
| 0.15          | 0.8102040816 |            2,813,380 |                 10 |         31.61% |

```{figure} ../img/pruning-nni-classification-comparison.*
---
name: pruning-nni-classification-comparison
alt: Pruning classification comparison
align: center
---

Model size, speed and quality comparison for NNI pruning
```

```{figure} ../img/pruning-nni-gpu-usage-comparison.*
---
name: pruning-nni-gpu-usage
alt: Pruning GPU usage comparison
align: center
---

Plot represents changes of GPU usage over time
```

```{figure} ../img/pruning-nni-gpu-mem-comparison.*
---
name: pruning-nni-gpu-mem
alt: Pruning GPU memory usage comparison
align: center
---

Plot represents changes of GPU RAM usage over time
```

