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

from abc import ABC, abstractmethod


class RunConfigGenerator(ABC):
    """
    An abstract class that parses analyzer config
    and generates RunConfigs
    """

    @abstractmethod
    def __iter__(self):
        """
        Allows using this object
        as an iterator
        """

        return self

    @abstractmethod
    def __next__(self):
        """
        Chooses the next set of run parameters

        Returns
        -------
        RunConfig
            Corresponding to a ModelConfig and some
            run parameters
        
        Raises
        ------
        StopIteration
            Reaching end of all parameters
        """
