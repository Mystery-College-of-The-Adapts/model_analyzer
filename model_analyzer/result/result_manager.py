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

import heapq
import logging

from .result_table import ResultTable
from .run_result import RunResult
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ResultManager:
    """
    This class provides methods to create, and add to 
    ResultTables. Each ResultTable holds results from
    multiple runs.
    """

    non_gpu_specific_headers = ['Model', 'Batch', 'Concurrency']
    gpu_specific_headers = ['Model', 'GPU ID', 'Batch', 'Concurrency']
    server_only_table_key = 'server_gpu_metrics'
    model_gpu_table_key = 'model_gpu_metrics'
    model_inference_table_key = 'model_inference_metrics'

    def __init__(self, config):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            the model analyzer config
        """

        self._config = config
        self._result_tables = {}
        self._current_run_results = {}
        self._result_comparator = None

        # Results are stored in a heap queue
        self._results = []

    def set_result_comparator(self, comparator):
        """
        Sets the ResultComparator for all the results
        this ResultManager will construct

        Parameters
        ----------
        comparator : ResultComparator
            the result comparator function object that can
            compare two results.
        """

        self._result_comparator = comparator

    def create_tables(self, gpu_specific_metrics, non_gpu_specific_metrics,
                      aggregation_tag):
        """
        Creates the tables to print hold, display, and write
        results

        Parameters
        ----------
        gpu_specific_metrics : list of RecordTypes
            The metrics that have a GPU id associated with them
        non_gpu_specific_metrics : list of RecordTypes
            The metrics that do not have a GPU id associated with them
        aggregation_tag : str
        """

        # Server only
        self._add_result_table(table_key=self.server_only_table_key,
                               title='Server Only',
                               headers=self.gpu_specific_headers,
                               metric_types=gpu_specific_metrics,
                               aggregation_tag=aggregation_tag)

        # Model Inference Tables
        self._add_result_table(table_key=self.model_gpu_table_key,
                               title='Models (GPU Metrics)',
                               headers=self.gpu_specific_headers,
                               metric_types=gpu_specific_metrics,
                               aggregation_tag=aggregation_tag)

        self._add_result_table(table_key=self.model_inference_table_key,
                               title='Models (Inference)',
                               headers=self.non_gpu_specific_headers,
                               metric_types=non_gpu_specific_metrics,
                               aggregation_tag=aggregation_tag)

    def init_result(self, run_config):
        """
        Initialize the RunResults
        for the current model run.
        There will be one result per table.

        Parameters
        ----------
        run_config : RunConfig
            The run config corresponding to the current 
            run
        """

        if len(self._result_tables) == 0:
            raise TritonModelAnalyzerException(
                "Cannot initialize results without tables")
        elif not self._result_comparator:
            raise TritonModelAnalyzerException(
                "Cannot initialize results without setting result comparator")

        # Create RunResult
        self._current_run_result = RunResult(
            run_config=run_config, comparator=self._result_comparator)

    def add_server_data(self, measurements, default_value):
        """
        Adds data to directly to the server only table

        Parameters
        ----------
        measurements : dict
            keys are gpu ids and values are lists of metric values
        default_value : val
            A value for those columns not applicable to standalone server
        """

        for gpu_id, metrics in measurements.items():
            data_row = ['triton-server', gpu_id, default_value, default_value]
            data_row += metrics
            self._result_tables[
                self.server_only_table_key].insert_row_by_index(data_row)

    def add_model_data(self, measurements, has_gpu_ids):
        """
        This function adds model inference
        measurements to the result, not directly
        to a table.

        Parameters
        ----------
        measurements : list
            The measurements from the metrics manager,
            actual values from the monitors
        has_gpu_ids : bool
            Whether these output metrics are gpu specific
            (for multi gpu settings)
        """

        # TODO filter out the measurements that fail constraints
        self._current_run_result.add_data(measurements=measurements,
                                          has_gpu_ids=has_gpu_ids)

    def complete_result(self):
        """
        Submit the current RunResults into
        the ResultTable
        """

        heapq.heappush(self._results, self._current_run_result)

    def compile_results(self):
        """
        The function called at the end of all runs 
        FOR A MODEL that compiles all result and 
        dumps the data into tables for exporting.
        """

        # Fill rows in descending order
        while self._results:
            next_best_result = heapq.heappop(self._results)
            self._compile_result(next_best_result)

    def _compile_result(self, result):
        """
        Puts the measurements in the 
        result into the tables.
        """

        gpu_data, non_gpu_rows = result.get_measurements()
        perf_configs = result.get_run_config().perf_analyzer_configs()

        for i in range(len(perf_configs)):
            model_name = perf_configs[i]['model-name']
            batch_size = perf_configs[i]['batch-size']
            concurrency = perf_configs[i]['concurrency-range']

            # Non GPU specific data
            inference_metrics = [model_name, batch_size, concurrency]
            inference_metrics += non_gpu_rows[i]
            self._result_tables[
                self.model_inference_table_key].insert_row_by_index(
                    row=inference_metrics)

            # GPU specific data
            for gpu_id, metrics in gpu_data[i].items():
                gpu_metrics = [model_name, gpu_id, batch_size, concurrency]
                gpu_metrics += metrics
                self._result_tables[
                    self.model_gpu_table_key].insert_row_by_index(
                        row=gpu_metrics)

    def get_all_tables(self):
        """
        Returns
        -------
        dict 
            table keys and ResultTables
        """

        return self._result_tables

    def get_server_table(self):
        """
        Returns
        -------
        ResultTable
            The table corresponding to server only
            data
        """

        return self._get_table(self.server_only_table_key)

    def get_model_tables(self):
        """
        Returns
        -------
        (ResultTable, ResultTable)
            The table corresponding to the model inference
            data
        """

        return self._get_table(self.model_gpu_table_key), self._get_table(
            self.model_inference_table_key)

    def _get_table(self, key):
        """
        Get a ResultTable by table key
        """

        if key not in self._result_tables:
            raise TritonModelAnalyzerException(
                f"Table with key '{key}' not found in ResultManager")
        return self._result_tables[key]

    def _add_result_table(self,
                          table_key,
                          title,
                          headers,
                          metric_types,
                          aggregation_tag='Max'):
        """
        Utility function that creates a table with column
        headers corresponding to perf_analyzer arguments
        and requested metrics. Also sets the result
        comparator for that table.
        """

        # Create headers
        table_headers = headers[:]
        for metric in metric_types:
            table_headers.append(metric.header(aggregation_tag + " "))
        self._result_tables[table_key] = ResultTable(headers=table_headers,
                                                     title=title)
