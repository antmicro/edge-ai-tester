#!/usr/bin/env python

"""
A script that generates report files based on Measurements JSON output.

It requires providing the report type and JSON file to extract data from.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.resources import reports
from kenning.core.drawing import time_series_plot
from kenning.core.drawing import draw_confusion_matrix
from kenning.core.drawing import recall_precision_curves
from kenning.core.drawing import recall_precision_gradients
from kenning.core.drawing import true_positive_iou_histogram
from kenning.core.drawing import true_positives_per_iou_range_histogram
from kenning.core.drawing import draw_plot
from kenning.utils import logger
from kenning.core.report import create_report_from_measurements
from kenning.utils.class_loader import get_command


log = logger.get_logger()


def performance_report(
        reportname: str,
        measurementsdata: Dict[str, List],
        imgdir: Path,
        reportpath: Path) -> str:
    """
    Creates performance section of the report.

    Parameters
    ----------
    reportname : str
        Name of the report
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class
    imgdir : Path
        Path to the directory for images
    reportpath : Path
        Path to the output report

    Returns
    -------
    str : content of the report in RST format
    """
    log.info('Running performance_report')

    inference_step = None
    if 'target_inference_step' in measurementsdata:
        log.info('Using target measurements for inference time')
        inference_step = 'target_inference_step'
    elif 'protocol_inference_step' in measurementsdata:
        log.info('Using protocol measurements for inference time')
        inference_step = 'protocol_inference_step'
    else:
        log.warning('No inference time measurements in the report')

    if inference_step:
        usepath = imgdir / f'{reportpath.stem}_inference_time.png'
        time_series_plot(
            str(usepath),
            f'Inference time for {reportname}',
            'Time', 's',
            'Inference time', 's',
            measurementsdata[f'{inference_step}_timestamp'],
            measurementsdata[inference_step],
            skipfirst=True)
        measurementsdata['inferencetimepath'] = str(usepath)
        measurementsdata['inferencetime'] = \
            measurementsdata[inference_step]

    if 'session_utilization_mem_percent' in measurementsdata:
        log.info('Using target measurements memory usage percentage')
        usepath = imgdir / f'{reportpath.stem}_cpu_memory_usage.png'
        time_series_plot(
            str(usepath),
            f'Memory usage for {reportname}',
            'Time', 's',
            'Memory usage', '%',
            measurementsdata['session_utilization_timestamp'],
            measurementsdata['session_utilization_mem_percent'])
        measurementsdata['memusagepath'] = str(usepath)
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_cpus_percent' in measurementsdata:
        log.info('Using target measurements CPU usage percentage')
        usepath = imgdir / f'{reportpath.stem}_cpu_usage.png'
        measurementsdata['session_utilization_cpus_percent_avg'] = [
            np.mean(cpus) for cpus in
            measurementsdata['session_utilization_cpus_percent']
        ]
        time_series_plot(
            str(usepath),
            f'Mean CPU usage for {reportname}',
            'Time', 's',
            'Mean CPU usage', '%',
            measurementsdata['session_utilization_timestamp'],
            measurementsdata['session_utilization_cpus_percent_avg'])
        measurementsdata['cpuusagepath'] = str(usepath)
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_gpu_mem_utilization' in measurementsdata:
        log.info('Using target measurements GPU memory usage percentage')
        usepath = imgdir / f'{reportpath.stem}_gpu_memory_usage.png'
        gpumemmetric = 'session_utilization_gpu_mem_utilization'
        if len(measurementsdata[gpumemmetric]) == 0:
            log.warning(
                'Incorrectly collected data for GPU memory utilization'
            )
        else:
            time_series_plot(
                str(usepath),
                f'GPU memory usage for {reportname}',
                'Time', 's',
                'GPU memory usage', 'MB',
                measurementsdata['session_utilization_gpu_timestamp'],
                measurementsdata[gpumemmetric])
            measurementsdata['gpumemusagepath'] = str(usepath)
    else:
        log.warning('No GPU memory usage measurements in the report')

    if 'session_utilization_gpu_utilization' in measurementsdata:
        log.info('Using target measurements GPU utilization')
        usepath = imgdir / f'{reportpath.stem}_gpu_usage.png'
        if len(measurementsdata['session_utilization_gpu_utilization']) == 0:
            log.warning('Incorrectly collected data for GPU utilization')
        else:
            time_series_plot(
                str(usepath),
                f'GPU Utilization for {reportname}',
                'Time', 's',
                'Utilization', '%',
                measurementsdata['session_utilization_gpu_timestamp'],
                measurementsdata['session_utilization_gpu_utilization'])
            measurementsdata['gpuusagepath'] = str(usepath)
    else:
        log.warning('No GPU utilization measurements in the report')

    with path(reports, 'performance.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def classification_report(
        reportname: str,
        measurementsdata: Dict[str, List],
        imgdir: Path,
        reportpath: Path):
    """
    Creates classification quality section of the report.

    Parameters
    ----------
    reportname : str
        Name of the report
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class
    imgdir : Path
        Path to the directory for images
    reportpath : Path
        Path to the output report

    Returns
    -------
    str : content of the report in RST format
    """
    log.info('Running classification report')

    if 'eval_confusion_matrix' not in measurementsdata:
        log.error('Confusion matrix not present for classification report')
        return ''
    log.info('Using confusion matrix')
    confusionpath = imgdir / f'{reportpath.stem}_confusion_matrix.png'
    draw_confusion_matrix(
        measurementsdata['eval_confusion_matrix'],
        str(confusionpath),
        'Confusion matrix',
        measurementsdata['class_names']
    )
    measurementsdata['confusionpath'] = str(confusionpath)
    with path(reports, 'classification.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def detection_report(
        reportname: str,
        measurementsdata: Dict[str, List],
        imgdir: Path,
        reportpath: Path) -> str:
    """
    Creates detection quality section of the report.

    Parameters
    ----------
    reportname : str
        Name of the report
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class
    imgdir : Path
        Path to the directory for images
    reportpath : Path
        Path to the output report

    Returns
    -------
    str : content of the report in RST format
    """

    from kenning.datasets.helpers.detection_and_segmentation import \
        compute_ap11, \
        get_recall_precision, \
        compute_map_per_threshold

    log.info('Running detection report')

    lines = get_recall_precision(measurementsdata, 0.5)

    aps = []
    for line in lines:
        aps.append(compute_ap11(line[0], line[1]))

    measurementsdata['mAP'] = np.mean(aps)

    curvepath = imgdir / f'{reportpath.stem}_recall_precision_curves.png'
    recall_precision_curves(
        str(curvepath),
        'Recall-Precision curves',
        lines,
        measurementsdata['class_names']
    )
    measurementsdata['curvepath'] = str(curvepath)

    gradientpath = imgdir / f'{reportpath.stem}_recall_precision_gradients.png'
    recall_precision_gradients(
        str(gradientpath),
        'Average precision plots',
        lines,
        measurementsdata['class_names'],
        aps
    )
    measurementsdata['gradientpath'] = str(gradientpath)

    tp_iou = []
    all_tp_ious = []

    for i in measurementsdata['class_names']:
        dets = measurementsdata[f'eval_det/{i}'] if f'eval_det/{i}' in measurementsdata else []  # noqa: E501
        det_tp_iou = [i[2] for i in dets if i[1]]
        if len(det_tp_iou) > 0:
            tp_iou.append(sum(det_tp_iou)/len(det_tp_iou))
            all_tp_ious.extend(det_tp_iou)
        else:
            tp_iou.append(0)
    tpioupath = imgdir / f'{reportpath.stem}_true_positive_iou_histogram.png'
    iouhistpath = imgdir / f'{reportpath.stem}_histogram_tp_iou_values.png'

    true_positive_iou_histogram(
        str(tpioupath),
        'Average True Positive IoU values',
        tp_iou,
        measurementsdata['class_names'],
    )
    measurementsdata['tpioupath'] = str(tpioupath)

    if len(all_tp_ious) > 0:
        true_positives_per_iou_range_histogram(
            str(iouhistpath),
            "Histogram of True Positive IoU values",
            all_tp_ious
        )
        measurementsdata['iouhistpath'] = str(iouhistpath)

    thresholds = np.arange(0.2, 1.05, 0.05)
    mapvalues = compute_map_per_threshold(measurementsdata, thresholds)

    mappath = imgdir / f'{reportpath.stem}_map.png'
    draw_plot(
        str(mappath),
        'mAP value change over objectness threshold values',
        'threshold',
        None,
        'mAP',
        None,
        [thresholds, mapvalues]
    )
    measurementsdata['mappath'] = str(mappath)
    measurementsdata['max_mAP'] = max(mapvalues)
    measurementsdata['max_mAP_index'] = thresholds[np.argmax(mapvalues)].round(2)  # noqa: E501

    with path(reports, 'detection.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def generate_report(
        reportname: str,
        data: Dict,
        imgdir: Path,
        report_types: List[str],
        rootdir: Path) -> str:
    """
    Generates an RST report based on Measurements data.

    The report is saved to the file in ``outputpath``.

    Parameters
    ----------
    reportname : str
        Name for the report
    data : Dict
        Data coming from the Measurements object, loaded i.e. from JSON file
    imgdir : Path
        Path to the directory where the report plots should be stored
    report_types : List[str]
        List of report types that define the project, i.e.
        performance, classification
    rootdir : Path
        When the report is a part of a larger RST document (i.e. Sphinx docs),
        the rootdir parameter defines thte root directory of the document.
        It is used to compute relative paths in the document's references.
    """

    outputpath = rootdir / "report.rst"
    reptypes = {
        'performance': performance_report,
        'classification': classification_report,
        'detection': detection_report
    }

    content = ''
    data['reportname'] = [reportname]
    for typ in report_types:
        content += reptypes[typ](reportname, data, imgdir, outputpath)

    with open(outputpath, 'w') as out:
        out.write(content)


def main(argv):
    command = get_command(argv)
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'measurements',
        help='Path to the JSON file with measurements',
        type=Path
    )
    parser.add_argument(
        'reportname',
        help='Name of the report',
        type=str
    )
    parser.add_argument(
        'outputdir',
        help='Path to root directory (report and generated images will be stored here)',  # noqa: E501
        type=Path
    )
    parser.add_argument(
        '--report-types',
        help='List of types that implement this report',
        nargs='+',
        required=True,
        type=str
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args = parser.parse_args(argv[1:])

    args.outputdir.mkdir(parents=True, exist_ok=True)
    root_dir = args.outputdir.absolute()
    img_dir = root_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(args.measurements, 'r') as measurements:
        measurementsdata = json.load(measurements)

    if 'build_cfg' in measurementsdata:
        measurementsdata['build_cfg'] = json.dumps(
            measurementsdata['build_cfg'],
            indent=4
        ).split('\n')

    if 'command' in measurementsdata:
        measurementsdata['command'] += [''] + command

    generate_report(
        args.reportname,
        measurementsdata,
        img_dir,
        args.report_types,
        root_dir
    )


if __name__ == '__main__':
    main(sys.argv)
