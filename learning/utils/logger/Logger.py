import json
import os

import wandb

from .EpisodicLogs import EpisodicLogs
from .PerIterationLogs import PerIterationLogs
from .TimeKeeper import TimeKeeper


class Logger:
    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        num_envs=1,
        episode_dt=0.1,
        total_iterations=100,
        device="cpu",
        log_dir=None,
    ):
        self.device = device

        self.reward_logs = EpisodicLogs(num_envs, episode_dt, device=device)

        self.iteration_logs = PerIterationLogs()

        self.iteration_counter = 0
        self.tot_iter = total_iterations

        self.timer = TimeKeeper()
        self.num_envs = num_envs
        self.step_counter = 0
        self.algorithm_logs = {}
        # Optional local per-iteration vitals sink (independent of wandb) —
        # one JSON object per line at <log_dir>/vitals.jsonl.
        self.log_dir = log_dir
        self.initialized = True

    def register_category(self, category, target, attribute_list):
        self.iteration_logs.register_items(category, target, attribute_list)

    def register_rewards(self, reward_names):
        self.reward_logs.add_buffer(reward_names)

    def log_rewards(self, rewards_dict):
        self.reward_logs.add_step(rewards_dict)

    def finish_step(self, dones):
        self.reward_logs.finish_step(dones)
        self.step_counter += 1

    def finish_iteration(self):
        self.iteration_counter += 1
        if wandb.run is not None:
            self.log_to_wandb()
        if self.log_dir is not None:
            self.log_to_file()
        return None

    @staticmethod
    def _to_scalar(val):
        # action_std / entropy come through as (grad-carrying) tensors, possibly
        # per-action; detach and reduce anything multi-element to its mean.
        # No try/except (fail-fast).
        if hasattr(val, "detach"):
            val = val.detach()
        if hasattr(val, "numel") and val.numel() != 1:
            return float(val.float().mean())
        return float(val)

    def collect_vitals(self):
        """Flat {metric: scalar} snapshot of this iteration (shared by the
        wandb and file sinks)."""
        record = {"iteration": self.iteration_counter}
        for key, val in self.reward_logs.get_average_rewards().items():
            record[f"rewards/{key}"] = self._to_scalar(val)
        for category in self.iteration_logs.logs.keys():
            for key, val in self.iteration_logs.get_all_logs(category).items():
                record[f"{category}/{key}"] = self._to_scalar(val)
        record["episode_time"] = float(self.reward_logs.get_average_time())
        record["steps_per_s"] = float(self.estimate_steps_per_second())
        record["t_iteration"] = float(self.timer.get_time("iteration"))
        record["t_collection"] = float(self.timer.get_time("collection"))
        record["t_learning"] = float(self.timer.get_time("learning"))
        return record

    def log_to_file(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "vitals.jsonl"), "a") as f:
            f.write(json.dumps(self.collect_vitals()) + "\n")

    def estimate_ETA(self, times=["runtime"], mode="total"):
        if mode == "total":
            total_time = sum(self.timer.get_time(part) for part in times)
            iter_time = total_time / self.iteration_counter
        elif mode == "iteration":
            # sum all the times in times
            iter_time = sum(self.timer.get_time(part) for part in times)
        else:
            iter_time = 0.0
        return iter_time * (self.tot_iter - self.iteration_counter)

    def format_seconds_to_hms(self, secs):
        minutes, seconds = divmod(int(secs), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"

    def estimate_steps_per_second(self):
        return (self.step_counter * self.num_envs / self.iteration_counter) / (
            self.timer.get_time("collection")
        )

    def print_to_terminal(self):
        width = 80
        pad = 35

        log_string = ""

        def format_log_entry(key, v=None, append=""):
            """Helper function to format a single log entry."""
            nonlocal log_string
            if v is None:
                log_string += f"{key:>{pad}}: {append}\n"
            else:
                log_string += f"{key:>{pad}}: {v:.2f} {append}\n"

        def separator(subtitle="", marker="-"):
            nonlocal log_string
            subtitle_length = len(subtitle)
            if subtitle_length > 0:
                subtitle_length += 1  # Add 1 for the space after subtitle

            dashes_each_side = (width - subtitle_length) // 2
            log_string += "\n"
            log_string += (
                f"{marker * dashes_each_side} {subtitle} {marker * dashes_each_side}\n"
            )

        separator(f"Iteration {self.iteration_counter}/{self.tot_iter}", "#")

        separator("Rewards")
        averages = self.reward_logs.get_average_rewards()

        for key, val in averages.items():
            format_log_entry(key, val)

        separator("Other Agent Metrics")
        format_log_entry("average episode time", self.reward_logs.get_average_time())

        separator("Algorithm Performance")
        for key, val in self.iteration_logs.get_all_logs("algorithm").items():
            format_log_entry(key, val)

        separator("Timings")
        format_log_entry("steps/s", self.estimate_steps_per_second())
        tot_t = self.timer.get_time("iteration")
        col_time = self.timer.get_time("collection")
        learn_time = self.timer.get_time("learning")
        time_string = f"(sim: {col_time:.2f}, learn:{learn_time:.2f})"
        format_log_entry("total time", tot_t, time_string)

        format_log_entry(
            "ETA", append=self.format_seconds_to_hms(self.estimate_ETA(["runtime"]))
        )
        print(log_string)

    def log_all_categories(self):
        for category in self.iteration_logs.logs.keys():
            self.iteration_logs.log(category)

    def log_to_wandb(self):
        def prepend_to_keys(prefix, dictionary):
            return {prefix + "/" + key: val for key, val in dictionary.items()}

        averages = prepend_to_keys("rewards", self.reward_logs.get_average_rewards())

        category_logs = {
            f"{category}/{key}": val
            for category in self.iteration_logs.logs.keys()
            for key, val in self.iteration_logs.get_all_logs(category).items()
        }

        wandb.log({**averages, **category_logs})

    def tic(self, category="default"):
        self.timer.tic(category)

    def toc(self, category="default"):
        self.timer.toc(category)

    def get_time(self, category="default"):
        return self.timer.get_time(category)

    def attach_torch_obj_to_wandb(self, obj_tuple, log_freq=100, log_graph=True):
        if wandb.run is None:
            return
        if type(obj_tuple) is not tuple:
            obj_tuple = (obj_tuple,)
        wandb.watch(obj_tuple, log_freq=log_freq, log_graph=log_graph)
