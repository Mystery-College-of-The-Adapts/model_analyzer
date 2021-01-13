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

import os


def parse_model_config(model_path):
    """
    Parses a config.pbtxt as a string

    Parameters
    ----------
    model_path : str
        The full path to the model's root directory

    Returns:
        a complex dict where keys are config fields
    """

    with open(os.path.join(model_path, "config.pbtxt"), 'r+') as f:
        config_str = f.read()

    model_config = {}
    parse_dict(config_str, model_config)
    return model_config


def parse_dict(config_str, output):
    """
    Recursive function to parse a dict,
    called when parse_model_config sees a "{"
    This will parse one dict.

    Parameters
    ----------
    config_str: str
        What remains of the config string to be parsed
    
    output :  list
        The list you are currently parsing
    
    Returns
    -------
    str
        Input string minus what was consumed
    """
    # now scan to next \n look for ":, [, or {"
    while True:
        # Strip the leading white space at start of dict
        config_str = config_str.lstrip()
        next_newline = config_str.find('\n')
        if next_newline <= 0:
            return config_str
        else:
            next_item = config_str[:next_newline]
            key_delimiter = next_item.find(':')
            list_delimiter = next_item.find('[')
            dict_delimiter = next_item.find('{')

        if key_delimiter > -1:
            next_val = next_item[key_delimiter + 1:].lstrip().replace('"', '')
            output[next_item[:key_delimiter]] = next_val
            config_str = config_str[next_newline + 1:]
        elif list_delimiter > -1:
            next_list = []
            config_str = parse_list(config_str[list_delimiter + 1:],
                                    next_list).lstrip()
            output[next_item[:list_delimiter].rstrip()] = next_list
        elif dict_delimiter > -1:
            next_dict = {}
            config_str = parse_dict(config_str[dict_delimiter + 1:],
                                    next_dict).lstrip()
            output[next_item[:dict_delimiter].rstrip()] = next_dict
        else:
            break

    return config_str


def parse_list(config_str, output):
    """
    Recursive function to parse a list,
    called when parse_model_config sees a "["
    This will parse one list.

    Parameters
    ----------
    config_str: str
        What remains of the config string to be parsed
    
    output :  list
        The list you are currently parsing

    Returns
    -------
    str
        Input string minus what was consumed
    """

    # now scan to next ',' or '{'
    while True:
        # Strip the leading white space at start of dict
        config_str = config_str.lstrip()
        if config_str[0] == ']':
            config_str = config_str[1:].lstrip()
            break
        elif config_str[0] == '{':
            next_dict = {}
            config_str = parse_dict(config_str.lstrip()[1:],
                                    next_dict).lstrip()[1:]
            output.append(next_dict)
        else:
            comma = config_str.find(',')
            end_list = config_str.find(']')

            # Check for case where its last element of list
            if (comma < 0) or (comma > end_list):
                delimiter = end_list
            else:
                delimiter = comma
            output.append(config_str[:delimiter].strip())
            config_str = config_str[delimiter + 1:]
    return config_str
