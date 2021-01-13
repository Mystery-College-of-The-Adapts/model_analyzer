# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class ResultComparator:
    """
    A function object that configurably
    computes a float score for each RunResult.
    """

    def __init__(self,
                 gpu_metric_types,
                 non_gpu_metric_types,
                 metric_priorities,
                 comparison_threshold_percent=1,
                 weights=None):
        """
        Parameters
        ----------
        gpu_metric_types : list of RecordTypes
            The types of measurements in the measurements
            list that have a GPU ID, in the order they appear
        non_gpu_metric_types : list of RecordTypes
            The types of measurements in the measurements
            list that do NOT have a GPU ID, in the order they appear
        metric_priorities : list of RecordTypes
            The priority of the types above (i.e. the order
            of comparison)
        comparison_threshold_percent : int
            The threshold within with two measurements are considered
            equal as a percentage of the first measurement
        weights : list of float
            The relative importance of the priorities with respect to others
            If None, then equal weighting (normal priority comparison)
        """

        # Index the non gpu metric types
        self._non_gpu_type_to_idx = {
            non_gpu_metric_types[i]: i
            for i in range(len(non_gpu_metric_types))
        }

        # Index the gpu metric types
        self._gpu_type_to_idx = {
            gpu_metric_types[i]: i
            for i in range(len(gpu_metric_types))
        }

        self._metric_priorities = metric_priorities
        self._comparison_threshold_factor = (comparison_threshold_percent *
                                             1.0) / 100

        # TODO implement a weighted comparison
        self._weights = weights

    def __call__(self, result1, result2):
        """
        Allows an instance of a ResultComparator
        to behave like a function

        Parameters
        ----------
        result1 : RunResult
            first result to be compared
        result2 : RunResult
            second result to be compared
        
        Returns
        -------
        int 
            0 
                if the results are determined
                to be the same within a threshold
            1
                if result1 > result2
            -1
                if result1 < result2
        """

        # For now, average over perf runs for given model_config
        avg_gpu_metrics1, avg_non_gpu_metrics1 = \
            self._average_measurements(result1)
        avg_gpu_metrics2, avg_non_gpu_metrics2 = \
            self._average_measurements(result2)

        for priority in self._metric_priorities:
            if priority in self._gpu_type_to_idx:
                # Get position in measurements of current priority's value
                metric_idx = self._gpu_type_to_idx[priority]
                threshold = self._comparison_threshold_factor * avg_non_gpu_metrics1[
                    metric_idx]
                value_diff = avg_non_gpu_metrics1[
                    metric_idx] - avg_non_gpu_metrics2[metric_idx]

                if value_diff > threshold:
                    return 1
                elif value_diff < -threshold:
                    return -1
            elif priority in self._gpu_type_to_idx:
                metric_idx = self._gpu_type_to_idx[priority]
                threshold = self._comparison_threshold_factor * avg_gpu_metrics1[
                    metric_idx]
                value_diff = avg_gpu_metrics1[metric_idx] - avg_gpu_metrics2[
                    metric_idx]

                if value_diff > threshold:
                    return 1
                elif value_diff < -threshold:
                    return -1
            else:
                raise TritonModelAnalyzerException(
                    f"Category unknown for objective : '{priority}'")
        return 0

    def _average_measurements(self, result):
        """
        Returns
        -------
        (list, list)
            A 2-tuple of average measurements, 
            The first is across non-gpu specific metrics
            The second is across gpu-specific measurements
        """

        # For the gpu_measurements we have a list of dicts of lists
        # Assumption here is that its okay to average over all GPUs over all perf runs
        gpu_measurements, non_gpu_measurements = result.get_measurements()
        gpu_rows = []
        for measurement in gpu_measurements:
            gpu_rows.append(measurement.values())

        return self._average_list(gpu_rows), self._average_list(
            non_gpu_measurements)

    def _average_list(self, row_list):
        """
        Averages a 2d list over the rows
        """

        if not row_list:
            return row_list
        else:
            N = len(row_list)
            d = len(row_list[0])
            avg = [0 for _ in range(d)]
            for i in range(d):
                avg[i] = (sum([row_list[j][i] for j in range(N)]) * 1.0) / N
            return avg
