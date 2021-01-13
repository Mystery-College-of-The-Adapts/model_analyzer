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

from .model_config_utils import parse_model_config
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ModelConfig:
    """
    A class that encapsulates
    all the metadata about
    a model.
    """

    def __init__(self, model_path, instance_group=None, dynamic_batching=None):
        """
        Parameters
        -------
        model_path : str
            The full path to this model directory
        instance_group : dict
            instance_group with KIND and count to initialize
            this model config
        dynamic_batching : dict
            to request dynamic batching for this model
        """

        self._model_path = model_path
        self._config = parse_model_config(model_path)
        if instance_group:
            self.set_instance_group(instance_group)
        if dynamic_batching:
            self.set_dynamic_batching(dynamic_batching)

    def name(self):
        """
        Returns
        -------
        str
            The name of this model
        """

        if 'name' not in self._config:
            raise TritonModelAnalyzerException(
                f"Model's name not found in ModelConfig at {self._model_path}")
        return self._config['name']

    def platform(self):
        """
        Returns
        -------
        str
            The platform name of this model
        """

        if 'platform' not in self._config:
            raise TritonModelAnalyzerException(
                "Model's platform not found in "
                f"ModelConfig at {self._model_path}")
        return self._config['platform']

    def max_batch_size(self):
        """
        Returns
        -------
        int
            The max batch size of the model
        """

        if 'max_batch_size' not in self._config:
            raise TritonModelAnalyzerException(
                "Model's max_batch_size not found in "
                f"ModelConfig at {self._model_path}")
        return int(self._config['max_batch_size'])

    def instance_group(self):
        """
        Returns
        -------
        dict
            The instance group which is a dict
            with keys like 'kind' and 'count'
        """
        if 'instance_group' not in self._config:
            raise TritonModelAnalyzerException(
                f"ModelConfig  at {self._model_path} does "
                "not contain an instance_group")
        return self._config['instance_group']

    def dynamic_batching(self):
        """
        Returns
        -------
        dict
            The dynamic batching parameters for this model
        """

        if 'dynamic_batching' not in self._config:
            raise TritonModelAnalyzerException(
                f"ModelConfig at {self._model_path} "
                "does not request dynamic_batching")
        return self._config['dynamic_batching']

    def set_instance_group(self, instance_group):
        """
        Sets the instance group field in a Triton
        Model Config

        Parameters
        ----------
        instance_group : dict
            with keys and values corresponding to an
            instance_group section of triton model
            config
        """

        self._config['instance_group'] = instance_group

    def set_dynamic_batching(self, dynamic_batching):
        """
        Sets the instance group field in a Triton
        Model Config

        Parameters
        ----------
        dynamic_batching : dict
            with keys and values corresponding to a
            dynamic_batching section of triton model
            config
        """

        self._config['dynamic_batching'] = dynamic_batching
