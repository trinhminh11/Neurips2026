from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleBatch:
    """Batch of transitions with trajectory context for reward computation."""

    trajectory: np.ndarray  # (B, k, n_states)     trajectory from states S_t..S_{t+k-1}
    actions: np.ndarray  # (B, n_actions)       action labels for the trajectory at S_t
    skills: np.ndarray  # (B) int64 skill labels for the trajectory at S_t
    dones: np.ndarray  # (B) bool indicating if the trajectory is done at S_t
    skill_starts: np.ndarray  # (B) bool indicating if the skill is started at S_t


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_states: int,
        n_actions: int,
        np_random: np.random.RandomState | None = None,
    ) -> None:
        self.capacity = capacity
        self.n_states = n_states
        self.n_actions = n_actions

        self.np_random = np_random if np_random is not None else np.random.RandomState()

        self._states: list[np.ndarray] = []  # (..., n_states)
        self._actions: list[np.ndarray] = []  # (..., n_actions) action labels
        self._skills: list[int] = []  # (..., 1) int64 skill labels
        self._dones: list[bool] = []  # (..., 1) bool indicating if the episode is done
        self._skill_starts: list[
            bool
        ] = []  # (..., 1) bool indicating if the skill is started

    def __len__(self) -> int:
        return len(self._states)

    def add(
        self,
        state: np.ndarray,  # (n_states) state at S_t
        action: np.ndarray,  # (n_actions) action label at S_t
        skill: int,  # skill label at S_t
        done: bool,  # bool indicating if the episode is done at S_t
        skill_start: bool,  # bool indicating if the skill is started at S_t
    ) -> None:
        self._states.append(np.asarray(state, dtype=np.float32).copy())
        self._actions.append(np.asarray(action, dtype=np.int64).copy())
        self._skills.append(np.asarray(skill, dtype=np.int64).copy())
        self._dones.append(np.asarray(done, dtype=bool).copy())
        self._skill_starts.append(np.asarray(skill_start, dtype=bool).copy())

        if len(self._states) > self.capacity:
            n_drop = len(self._states) - self.capacity
            del self._states[:n_drop]
            del self._actions[:n_drop]
            del self._skills[:n_drop]
            del self._dones[:n_drop]
            del self._skill_starts[:n_drop]

    def ready(self, batch_size: int) -> bool:
        if len(self._states) < batch_size:
            return False
        return True

    def sample(self, batch_size: int, k: int) -> SampleBatch:
        # firt, detect where done is True, get B random indices from done_idx - k and :-k
        done_idx = np.where(self._dones)[0]  # get indices where done is True

        # why ignore this: because we will get states from S_t..S_{t+k-1} and S_{t+k}, but if episode terminated at S_{t+i}, then S_{t+i+1} is not valid so we ignore from done_idx to done_idx-k+1
        ignored_idx = np.concatenate(
            [done_idx - k + 1 + i for i in range(k)]
        )  # get indices to ignore
        ignored_idx = np.concatenate(
            [ignored_idx, np.arange(len(self._states) - k, len(self._states))]
        )  # also ignore the last k states
        ignored_idx = np.sort(
            np.unique(ignored_idx)
        )  # sort and unique, sort is optional but good for debugging

        indices = np.setdiff1d(
            np.arange(len(self._states)), ignored_idx
        )  # get indices to sample from

        if len(indices) < batch_size:
            return None

        base_indices = self.np_random.choice(
            indices, size=batch_size, replace=False
        )  # the batch indices to sample from
        _reshaped_base_indices = base_indices.reshape(-1, 1)
        state_indices = np.concatenate(
            [_reshaped_base_indices + i for i in range(k + 1)], axis=1
        ).ravel()  # the state indices to sample from, that include S_t..S_{t+k-1} and S_{t+k} (that why range(k+1))

        dones = np.asarray([self._dones[idx] for idx in base_indices])
        actions = np.asarray([self._actions[idx] for idx in base_indices])
        skills = np.asarray([self._skills[idx] for idx in base_indices])
        skill_starts = np.asarray([self._skill_starts[idx] for idx in base_indices])

        trajectory = np.asarray([self._states[idx] for idx in state_indices]).reshape(
            batch_size, k + 1, self.n_states
        )  # the trajectory to sample from, that include S_t..S_{t+k-1} and S_{t+k}

        return SampleBatch(trajectory, actions, skills, dones, skill_starts)


def main():
    buffer = ReplayBuffer(capacity=1000, n_states=10, n_actions=10)

    skill = 3
    skill_start = False

    for i in range(20):
        state = np.random.rand(10)
        action = np.random.randint(0, 10, size=(2,))
        done = False
        buffer.add(state, action, skill, done, skill_start)

    state = np.random.rand(10)
    action = np.random.randint(0, 10, size=(2,))
    done = True
    buffer.add(state, action, skill, done, skill_start)

    for i in range(15):
        state = np.random.rand(10)
        action = np.random.randint(0, 10, size=(2,))
        done = False
        buffer.add(state, action, skill, done, skill_start)

    state = np.random.rand(10)
    action = np.random.randint(0, 10, size=(2,))
    done = True
    buffer.add(state, action, skill, done, skill_start)

    for i in range(25):
        state = np.random.rand(10)
        action = np.random.randint(0, 10, size=(2,))
        done = False
        buffer.add(state, action, skill, done, skill_start)

    state = np.random.rand(10)
    action = np.random.randint(0, 10, size=(2,))
    done = True
    buffer.add(state, action, skill, done, skill_start)

    batch = buffer.sample(batch_size=8, k = 16)

    print(batch.trajectory.shape, batch.actions.shape, batch.skills.shape, batch.dones.shape, batch.skill_starts.shape)


if __name__ == "__main__":
    main()
