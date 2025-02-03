# Deep Learning deployment stack

This chapter lists and describes typical actions performed on deep learning models before deployment on target devices.

## From training to deployment

A deep learning application deployed on IoT devices usually goes through the following process:

* a dataset is prepared for a deep learning process,
* evaluation metrics are specified based on a given dataset and outputs,
* data in the dataset undergoes analysis, data loaders that perform the preprocessing are implemented,
* the deep learning model is either designed from scratch or a baseline is selected from a wide selection of existing pre-trained models for a given deep learning application (classification, detection, semantic segmentation, instance segmentation, etc.) and adjusted to a particular use case,
* a loss function and a learning algorithm are specified along with the deep learning model,
* the model is trained, evaluated and improved,
* the model is compiled to a representation that is applicable to a given target,
* the model is executed on a target device.

## Dataset preparation

If a model is not available or it is trained for a different use case, the model needs to be trained or re-trained.

Each model requires a dataset - a set of sample inputs (audio signals, images, video sequences, OCT images, other sensors) and, usually, also outputs (association to class or classes, object location, object mask, input description).
Datasets are usually split into the following categories:

* training dataset - the largest subset that is used to train a model,
* validation dataset - a relatively small set that is used to verify model performance after each training epoch (the metrics and loss function values show if any overfitting occurred during the training process),
* test dataset - the subset that acts as the final evaluation of a trained model.

It is required that the test dataset and the training dataset are mutually exclusive, so that the evaluation results are not biased in any way.

Datasets can be either designed from scratch or found in e.g.:

* [Kaggle datasets](https://www.kaggle.com),
* [Google Dataset Search](https://datasetsearch.research.google.com),
* [Dataset list](https://www.datasetlist.com),
* Universities' pages,
* [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html),
* [Common Voice Dataset](https://commonvoice.mozilla.org/en).

## Model preparation and training

Currently, the most popular approach is to find an existing model that fits a given problem and perform transfer learning to adapt the model to the requirements.
In transfer learning, the existing model's final layers are slightly modified to adapt to a new problem. These updated final layers of the model are trained using the training dataset.
Finally, some additional layers are unfrozen and the training is performed on a larger number of parameters at a very small learning rate - this process is called fine-tuning.

Transfer learning provides a better starting point for the training process, allows to train a correctly performing model with smaller datasets and reduces the time required to train a model.
The intuition behind this is that there are multiple common features between various objects in real-life environments, and the features learned from one deep learning scenario can be then reused in another scenario.

Once a model is selected, it requires adequate data input preprocessing in order to perform valid training.
The input data should be normalized and resized to fit input tensor requirements.
In case of the training dataset, especially if it is quite small, applying reasonable data augmentations like random brightness, contrast, cropping, jitters or rotations can significantly improve the training process and prevent the network from overfitting.

In the end, a proper training procedure needs to be specified.
This step includes:

* loss function specification for the model.
  Some weights regularizations can be specified, along with the loss function, to reduce the chance of overfitting
* optimizer specification (like Adam, Adagrad).
  This involves setting hyperparameters properly or adding schedules and automated routines to set those hyperparameters (i.e. scheduling the learning rate value, or using LR-Finder to set the proper learning rate for the scenario)
* number of epochs specification or scheduling, e.g. early stopping can be introduced.
* providing some routines for quality metrics measurements
* providing some routines for saving intermediate models during training (periodically, or the best model according to a particular quality measure)

## Model optimization

A successfully trained model may require some optimizations in order to run on given IoT hardware.
The optimizations may involve precision of weights, computational representation, or model structure.

Models are usually trained with FP32 precision or mixed precision (FP32 + FP16, depending on the operator).
Some targets, on the other hand, may significantly benefit from changing the precision from FP32 to FP16, INT8 or INT4.
The optimizations here are straightforward for the FP16 precision, but the integer-based quantizations require dataset calibration to reduce precision without a significant loss in a model's quality.

Other optimizations change the computational representation of the model by e.g. layer fusion or specialized operators for convolutions of a particular shape, among others.

In the end, there are algorithmic optimizations that change the entire model structure, like weights pruning, conditional computation, model distillation (the current model acts as a teacher that is supposed to improve the quality of a much smaller model).

If these model optimizations are applied, the optimized models should be evaluated using the same metrics as the original model.
This is required in order to find any drops in quality.

## Model compilation and deployment

Deep learning compilers can transform model representation to:

* a source code for a different programming language, e.g. [Halide](https://halide-lang.org), C, C++, Java, that can be later used on a given target,
* a machine code utilizing available hardware accelerators with e.g. OpenGL, OpenCL, CUDA, TensorRT, ROCm libraries,
* an FPGA bitstream,
* other targets.

Those compiled models are optimized to perform as efficiently as possible on given target hardware.

In the final step, the models are deployed on a hardware device.
