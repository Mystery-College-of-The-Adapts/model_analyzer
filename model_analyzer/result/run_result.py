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

from functools import total_ordering


@total_ordering
class RunResult:
    """
    A class that represents the result of
    a single run. This RunResult belongs
    to a particular ResultTable
    """

    def __init__(self, run_config, comparator):
        """
        Parameters
        ----------
        run_config : RunConfig
            The model config corresponding with the current
            RunResult
        comparator : callable
            A callable that receives two results and 
            returns 1 if the first is better than 
            the second, 0 if they are equal and -1 
            otherwise
        """

        self._run_config = run_config
        self._comparator = comparator
        self._non_gpu_specific_measurements = []
        self._gpu_specific_measurements = []

    def add_data(self, measurements, has_gpu_ids):
        """
        This function adds model inference
        measurements to the result

        Parameters
        ----------
        measurements : list
            The measurements from the metrics manager,
            actual values from the monitors
        has_gpu_ids : bool
            Whether these output metrics are gpu specific
            (for multi gpu settings)
        """

        if has_gpu_ids:
            self._gpu_specific_measurements.append(measurements)
        else:
            self._non_gpu_specific_measurements.append(measurements)

    def get_measurements(self):
        """
        Returns
        -------
        (list, list)
            gpu_specific, and non gpu specific measurements
            respectively.
        """

        return (self._gpu_specific_measurements,
                self._non_gpu_specific_measurements)

    def get_run_config(self):
        """
        Returns
        -------
        RunConfig
            returns the run_config associated with this
            RunResult
        """

        return self._run_config

    def __eq__(self, other):
        """ 
        Checks for the equality of this and
        another RunResult
        """

        return (self._comparator(self, other) == 0)

    def __gt__(self, other):
        """
        Checks whether this RunResult is better
        than other
        """

        return (self._comparator(self, other) == 1)
