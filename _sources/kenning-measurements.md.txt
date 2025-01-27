# Kenning measurements

Kenning measurements are a set of information describing the compilation and evaluation process happening in Kenning.

They contain such information as:

* classes used to construct the optimization/runtime pipeline, along with their parameters,
* the JSON scenario used in the run,
* the command used to run the scenario,
* versions of the Python modules used,
* performance measurements, such as CPU usage, GPU usage,
* quality measurements, such as predictions, ground truth, confusion matrix

All information is stored in JSON format.

## Performance metrics

While quality measurements are problem-specific (collected in the `evaluate` method of the [](dataset-api) class), performance metrics are common across devices and applications.

Metrics are collected with a certain prefix `<prefix>`, indicating the scope of computations.
There are:

* `<prefix>_timestamp` - gives a timestamp for measurement collection in seconds.
* `<prefix>_cpus_percent` - gives per-core CPU utilization in % in a form of a list of lists.
  They are % of per-CPU usages for every timestamp.
* `<prefix>_mem_percent` - gives overall memory usage in %.
* `<prefix>_gpu_utilization` - gives overall GPU utilization in % (only works on platforms with NVIDIA GPUs and NVIDIA Jetson embedded devices).
* `<prefix>_gpu_mem_utilization` - gives GPU memory utilization in % (only works on platforms with NVIDIA GPUs and NVIDIA Jetson embedded devices).
