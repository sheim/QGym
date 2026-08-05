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
import platform
from datetime import datetime
from typing import Tuple

from learning.runners import get_runner_class
from learning.utils import set_discount_from_horizon

from gym import GYM_ROOT_DIR
from .helpers import class_to_dict, get_load_path, set_seed


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple:
        env_cfg = self.env_cfgs[name]
        train_cfg = self.train_cfgs[name]
        return env_cfg, train_cfg

    def set_log_dir_name(self, train_cfg, log_root="default"):
        if log_root == "default":
            log_root = os.path.join(
                GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
            )
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name,
            )
        elif log_root is None:
            log_dir = None
        train_cfg.log_dir = log_dir

    def convert_frequencies_to_params(self, env_cfg, train_cfg):
        self.set_control_and_sim_dt(env_cfg, train_cfg)
        self.set_discount_rates(train_cfg, env_cfg.control.ctrl_dt)

    def set_control_and_sim_dt(self, env_cfg, train_cfg):
        env_cfg.control.decimation = int(
            env_cfg.control.desired_sim_frequency / env_cfg.control.ctrl_frequency
        )
        env_cfg.control.ctrl_dt = 1.0 / env_cfg.control.ctrl_frequency
        env_cfg.sim_dt = env_cfg.control.ctrl_dt / env_cfg.control.decimation
        if env_cfg.sim_dt != 1.0 / env_cfg.control.desired_sim_frequency:
            print(
                f"****** Simulation dt adjusted from "
                f"{1.0 / env_cfg.control.desired_sim_frequency}"
                f" to {env_cfg.sim_dt}."
            )

        if not hasattr(train_cfg.actor, "frequency"):
            train_cfg.actor.frequency = env_cfg.control.ctrl_frequency

    def set_discount_rates(self, train_cfg, dt):
        if hasattr(train_cfg.algorithm, "discount_horizon"):
            hrzn = train_cfg.algorithm.discount_horizon
            train_cfg.algorithm.gamma = set_discount_from_horizon(dt, hrzn)

        if hasattr(train_cfg.algorithm, "GAE_bootstrap_horizon"):
            hrzn = train_cfg.algorithm.GAE_bootstrap_horizon
            train_cfg.algorithm.lam = set_discount_from_horizon(dt, hrzn)

    def make_env(
        self,
        name: str,
        env_cfg,
        device: str = "cpu",
        headless: bool = True,
        backend: str = "mujoco",
    ):
        """Instantiate a task using a supported physics backend.

        ``backend="mujoco"`` selects MuJocoWarpBackend or MuJocoCPUBackend
        based on device; ``backend="vsim"`` selects VSimBackend (CUDA-only).
        """
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        set_seed(env_cfg.seed)
        backend = select_backend(env_cfg, device, backend)
        env = task_class(
            cfg=env_cfg,
            device=device,
            headless=headless,
            backend=backend,
        )
        return env

    def make_alg_runner(self, env, train_cfg):
        train_cfg_dict = class_to_dict(train_cfg)
        runner_class = get_runner_class(train_cfg.runner_class_name)
        runner = runner_class(env, train_cfg_dict, train_cfg.runner.device)
        # * save resume path before creating a new log_dir
        if train_cfg.runner.resume:
            resume_path = get_load_path(
                name=train_cfg.runner.experiment_name,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner


def select_backend(cfg, device: str, backend: str = "mujoco"):
    """Choose a physics backend. Fail-fast: no silent fallbacks.

    backend="mujoco": cuda → MuJocoWarpBackend, otherwise MuJocoCPUBackend.
    backend="vsim":   VSimBackend (CUDA-only; vlearn must be installed and
                      the process started with .env.vsim — see that file).
    """
    if backend == "vsim":
        if not device.startswith("cuda"):
            raise RuntimeError(f"backend='vsim' is CUDA-only, got device={device!r}")
        from gym.envs.base.vsim_backend import VSimBackend

        return VSimBackend()
    if backend != "mujoco":
        raise ValueError(f"unknown backend {backend!r} (mujoco | vsim)")
    if device.startswith("cuda"):
        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

        return MuJocoWarpBackend()
    if platform.system() == "Darwin":
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

        return MuJocoCPUBackend()
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    return MuJocoCPUBackend()


# make global task registry
task_registry = TaskRegistry()
