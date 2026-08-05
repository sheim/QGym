# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import numpy as np
import random
import torch

from gym import GYM_ROOT_DIR
from gym.utils.torch_quat import to_torch


def class_to_dict(obj, torch_device=None) -> dict:
    """
    Convert a multi-level class (config) into a dict of dicts.
    Keyword arguments:
    torch_device (opt): if specified, converts each element to a torch tensor.
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        if torch_device:
            result[key] = to_torch(element, device=torch_device)
        else:
            result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_load_path(name, load_run=-1, checkpoint=-1):
    root = os.path.join(GYM_ROOT_DIR, "logs", name)
    run_path = select_run(root, load_run)
    model_name = select_model(run_path, checkpoint)
    load_path = os.path.join(run_path, model_name)
    return load_path


def select_run(root, load_run):
    try:
        runs = sorted(
            os.listdir(root),
            key=lambda x: os.path.getctime(os.path.join(root, x)),
        )
        if "exported" in runs:
            runs.remove("exported")
        if "videos" in runs:
            runs.remove("videos")
        last_run = os.path.join(root, runs[-1])
    except:  # ! no bare excepts!!!
        raise ValueError("No runs in this directory: " + root)

    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    return load_run


def select_model(load_run, checkpoint):
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return model


def randomize_episode_counters(env):
    env.episode_length_buf = torch.randint_like(
        env.episode_length_buf,
        high=int(env.max_episode_length),
    )
