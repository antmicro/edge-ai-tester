# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Union

import numpy as np

from kenning.utils import logger

log = logger.get_logger()
EPS = 1e-8


def accuracy(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes accuracy of the classifier based on confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def mean_precision(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes mean precision for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.mean(
        np.array(confusion_matrix).diagonal() /
        (np.sum(confusion_matrix, axis=0)+EPS)
    )


def mean_sensitivity(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes mean sensitivity for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.mean(
        np.array(confusion_matrix).diagonal() /
        (np.sum(confusion_matrix, axis=1)+EPS)
    )


def g_mean(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes g-mean metric for the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.float_power(np.prod(
        np.array(confusion_matrix).diagonal() /
        (np.sum(confusion_matrix, axis=1)+EPS)
    ), 1.0 / np.array(confusion_matrix).shape[0])


def compute_performance_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes performance metrics based on `measurementsdata` argument.
    If there is no performance metrics returns an empty dictionary.

    Computes mean, standard deviation and median for keys:

    * target_inference_step or protocol_inference_step
    * session_utilization_mem_percent
    * session_utilization_cpus_percent
    * session_utilization_gpu_mem_utilization
    * session_utilization_gpu_utilization

    Those metrics are stored as <name_of_key>_<mean|std|median>

    Additionaly computes time of first inference step
    as `inferencetime_first` and average utilization
    of all cpus used as `session_utilization_cpus_percent_avg` key.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """
    computed_metrics = {}

    def compute_metrics(metric_name: str, metric_value: Optional[Dict] = None):
        """
        Evaluates and saves metric in `operations` dictionary.

        Parameters
        ----------
        metric_name : str
            Name that is used to save matric evaluation
        metric_value : Optional[Dict]
            Values that are used to evaluate the metric
            If it is none then `measurementsdata[metric_name]` is used.
        """
        if not metric_value:
            metric_value = measurementsdata[metric_name]
        operations = {
            'mean': np.mean,
            'std': np.std,
            'median': np.median,
        }
        for op_name, op in operations.items():
            computed_metrics[f'{metric_name}_{op_name}'] = op(metric_value)

    # inferencetime
    inference_step = None
    if 'target_inference_step' in measurementsdata:
        inference_step = 'target_inference_step'
    elif 'protocol_inference_step' in measurementsdata:
        inference_step = 'protocol_inference_step'

    if inference_step:
        compute_metrics('inferencetime', measurementsdata[inference_step])

    # mem_percent
    if 'session_utilization_mem_percent' in measurementsdata:
        compute_metrics(
            'session_utilization_mem_percent',
            measurementsdata['session_utilization_mem_percent']
        )

    # cpus_percent
    if 'session_utilization_cpus_percent' in measurementsdata:
        cpus_percent_avg = [
            np.mean(cpus) for cpus in
            measurementsdata['session_utilization_cpus_percent']
        ]
        computed_metrics['session_utilization_cpus_percent_avg'] = cpus_percent_avg  # noqa: E501
        compute_metrics('session_utilization_cpus_percent_avg', cpus_percent_avg)  # noqa: E501

    # gpu_mem
    if 'session_utilization_gpu_mem_utilization' in measurementsdata:
        compute_metrics('session_utilization_gpu_mem_utilization')

    # gpu
    if 'session_utilization_gpu_utilization' in measurementsdata:
        compute_metrics('session_utilization_gpu_utilization')

    return computed_metrics


def compute_classification_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes classification metrics based on `measurementsdata` argument.
    If there is no classification metrics returns an empty dictionary.

    Computes accuracy, top 5 accuracy, precision, sensitivity and g mean of
    passed confusion matrix stored as `eval_confusion_matrix`.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """

    # If confusion matrix is not present in the measurementsdata, then
    # classification metrics can not be calculated.
    if 'eval_confusion_matrix' in measurementsdata:
        confusion_matrix = np.asarray(
            measurementsdata['eval_confusion_matrix'])
        confusion_matrix[np.isnan(confusion_matrix)] = 0.
        metrics = {
            'accuracy': accuracy(confusion_matrix),
            'mean_precision': mean_precision(confusion_matrix),
            'mean_sensitivity': mean_sensitivity(confusion_matrix),
            'g_mean': g_mean(confusion_matrix),
        }
        if 'top_5_count' in measurementsdata.keys():
            metrics['top_5_accuracy'] = \
                measurementsdata['top_5_count'] / measurementsdata['total']
        return metrics
    return {}


def compute_detection_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes detection metrics based on `measurementsdata` argument.
    If there is no detection metrics returns an empty dictionary.

    Computes mAP values.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """
    from kenning.datasets.helpers.detection_and_segmentation import \
        compute_map_per_threshold

    # If ground truths count is not present in the measurementsdata, then
    # mAP metric can not be calculated.
    if any(
            [key.startswith('eval_gtcount') for key in measurementsdata.keys()]
    ):
        return {
            'mAP': compute_map_per_threshold(measurementsdata, [0.0])[0]
        }
    return {}


def compute_renode_metrics(measurementsdata: List[Dict]) -> Dict:
    """
    Computes Renode metrics based on `measurementsdata` argument.
    If there is no Renode metrics returns an empty dictionary.

    Computes instructions counter for all opcodes and for V-Extension opcodes.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """
    if not any(('opcode_counters' in data for data in measurementsdata)):
        return {}

    # retrieve all opcodes with non zero counters
    all_opcodes = set()
    for data in measurementsdata:
        for opcode, counter in data['opcode_counters'].items():
            if counter > 0:
                all_opcodes.add(opcode)

    # retrieve counters
    opcode_counters = []

    for opcode in all_opcodes:
        counters = [opcode]
        for data in measurementsdata:
            counters.append(data['opcode_counters'].get(opcode, 0))
        opcode_counters.append(counters)

    opcode_counters.sort(key=lambda x: (sum(x[1:]), x[0]), reverse=True)

    vector_opcode_counters = [
        counters for counters in opcode_counters if counters[0][0] == 'v'
    ]

    ret = {}
    if len(opcode_counters):
        ret['sorted_opcode_counters'] = {}
        transposed = list(zip(*opcode_counters))
        ret['sorted_opcode_counters']['opcodes'] = transposed[0]
        if len(measurementsdata) == 1:
            ret['sorted_opcode_counters']['counters'] = {
                'counters': transposed[1]
            }
        else:
            ret['sorted_opcode_counters']['counters'] = {
                measurementsdata[i]['modelname']: transposed[i + 1]
                for i in range(len(measurementsdata))
            }

    if len(vector_opcode_counters):
        ret['sorted_vector_opcode_counters'] = {}
        transposed = list(zip(*vector_opcode_counters))
        ret['sorted_vector_opcode_counters']['opcodes'] = transposed[0]
        if len(measurementsdata) == 1:
            ret['sorted_vector_opcode_counters']['counters'] = {
                'counters': transposed[1]
            }
        else:
            ret['sorted_vector_opcode_counters']['counters'] = {
                measurementsdata[i]['modelname']: transposed[i + 1]
                for i in range(len(measurementsdata))
            }

    return ret
