"""Randomly proposed environments, revisited using their latest WM learning progress."""

from copy import deepcopy
import math

import numpy as np


class RandomLPEnvironmentPool:
    def __init__(self, capacity, new_fraction, uniform_mix, seed):
        if int(capacity) != capacity or capacity < 1:
            raise ValueError("random_lp.capacity must be a positive integer")
        if not 0 < new_fraction <= 1 or not 0 <= uniform_mix <= 1:
            raise ValueError("random_lp requires new_fraction in (0, 1] and uniform_mix in [0, 1]")
        self.capacity = int(capacity)
        self.new_fraction = float(new_fraction)
        self.uniform_mix = float(uniform_mix)
        self.rng = np.random.default_rng(seed)
        self.entries = []
        self.next_id = 0
        self.pending = None

    def mix(self, candidates, iteration):
        """Replace the last batch slots with distinct pool entries; do no rollouts here."""
        if self.pending is not None:
            raise RuntimeError("Random LP batch must be scored before selecting the next batch")
        batch_size = len(candidates)
        replay_count = min(len(self.entries), batch_size - math.ceil(batch_size * self.new_fraction))
        new_count = batch_size - replay_count
        batch, records = [], []
        for candidate in candidates[:new_count]:
            batch.append(deepcopy(candidate))
            records.append({
                "environment_id": self.next_id, "is_replay": False,
                "sampling_probability": None, "score_age": 0, "previous_lp": None,
            })
            self.next_id += 1

        remaining = list(range(len(self.entries)))
        for _ in range(replay_count):
            weights = np.asarray([max(self.entries[i]["lp"], 0.0) for i in remaining], dtype=np.float64)
            probabilities = np.full(len(remaining), 1.0 / len(remaining))
            if weights.sum() > 0:
                probabilities = self.uniform_mix * probabilities + (1 - self.uniform_mix) * weights / weights.sum()
            selected = int(self.rng.choice(len(remaining), p=probabilities))
            entry = self.entries[remaining.pop(selected)]
            batch.append(deepcopy(entry["candidate"]))
            records.append({
                "environment_id": entry["id"], "is_replay": True,
                "sampling_probability": float(probabilities[selected]),
                "score_age": int(iteration - entry["last_iteration"]),
                "previous_lp": entry["lp"],
            })
        self.pending = (batch, records, int(iteration))
        return batch

    def update(self, pre, post, valid, scorable=None):
        if self.pending is None:
            raise RuntimeError("Random LP scores require a selected environment batch")
        batch, records, iteration = self.pending
        pre, post = np.asarray(pre, dtype=np.float64), np.asarray(post, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        scorable = valid.copy() if scorable is None else np.asarray(scorable, dtype=bool) & valid
        if any(values.shape != (len(batch),) for values in (pre, post, valid, scorable)):
            raise ValueError("Random LP requires one pre/post loss and validity flag per environment")
        if not np.isfinite(pre[scorable]).all() or not np.isfinite(post[scorable]).all():
            raise ValueError("Random LP requires finite pre/post losses for every valid probe")
        by_id = {entry["id"]: entry for entry in self.entries}
        for index, (candidate, record) in enumerate(zip(batch, records)):
            lp = float(pre[index] - post[index]) if scorable[index] else 0.0
            if record["is_replay"]:
                entry = by_id[record["environment_id"]]
                entry.update(lp=lp, last_iteration=iteration)
            elif valid[index]:
                self.entries.append({
                    "id": record["environment_id"], "candidate": candidate,
                    "lp": lp, "last_iteration": iteration,
                })
            record.update(
                map_index=index, valid=bool(valid[index]), lp_valid=bool(scorable[index]),
                pre_changed_focal_loss=float(pre[index]) if scorable[index] else None,
                post_changed_focal_loss=float(post[index]) if scorable[index] else None,
                learning_progress=lp if scorable[index] else None,
            )
        self.entries = self.entries[-self.capacity:]
        for record in records:
            record["pool_size"] = len(self.entries)
        self.pending = None
        return records

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("Random LP state can only be saved after scoring the batch")
        return deepcopy({
            "capacity": self.capacity, "new_fraction": self.new_fraction,
            "uniform_mix": self.uniform_mix, "entries": self.entries,
            "next_id": self.next_id, "rng": self.rng.bit_generator.state,
        })

    def load_state_dict(self, state):
        for key in ("capacity", "new_fraction", "uniform_mix"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Random LP resume differs in {key}")
        self.entries = deepcopy(state["entries"])
        self.next_id = int(state["next_id"])
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.pending = None
