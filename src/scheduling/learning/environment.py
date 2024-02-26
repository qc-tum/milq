from enum import Enum
from typing import Any

from gymnasium import spaces
import gymnasium as gym
from gymnasium.core import RenderFrame
import numpy as np

from src.common import CircuitJob
from src.provider import Accelerator
from src.tools import assemble_job
from .action_space import ActionSpace
from ..heuristics.types import Machine, Schedule, Bucket, is_feasible
from ..heuristics.select import evaluate_solution
from ..heuristics.initialize import _convert_to_jobs


class Actions(Enum):
    """The posible actions and their corresponding integer values.
    CUT: 0, COMBINE: 1, MOVE: 2, SWAP: 3, TERMINATE: 4
    """

    CUT_CIRCUIT = 0
    COMBINE_CIRCUIT = 1
    MOVE_CIRCUIT = 2
    SWAP_CIRCUITS = 3
    TERMINATE = 4


class SchedulingEnv(gym.Env):
    """
    TODO
    - check if it makes sense to allow agent to specify where to put, same for other actions
    - check validity of merging and cutting circuits -> need to keep track somehow
    """

    def __init__(
        self,
        accelerators: list[Accelerator],
        circuits: list[CircuitJob],  # TODO generate CircuitJob from QuantumCircuit
        max_steps: int = 1000,  # max number of steps in an episode
        penalty: float = 5.0,  # penalty for invalid cuts
    ):
        super().__init__()
        self.penalty = penalty
        self.steps = 0
        self.max_steps = max_steps
        self.accelerators = accelerators
        self.circuits: list[CircuitJob] = circuits

        self._schedule = Schedule(
            [
                Machine(accelerator.qubits, str(accelerator.uuid), [], np.inf)
                for accelerator in accelerators
            ],
            np.inf,
        )  # Initialize with empty schedules for each device
        for circuit in self.circuits:
            choice = np.random.choice(len(self._schedule.machines))
            self._schedule.machines[choice].buckets.append(Bucket([circuit]))
        # Define the action and observation spaces
        self._action_space = ActionSpace(circuits, self._schedule)
        self.action_space = spaces.flatten_space(self._action_space)
        self.observation_space = spaces.Dict(
            {
                "makespan": spaces.Box(0, np.inf, dtype=np.float64),
                "noise": spaces.Box(0, np.inf, dtype=np.float64),
            }
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        penalty = 1.0  # multiplcative penalty for invalid cuts
        # Perform the specified action and update the schedule
        dict_action = self._unflatten_action(action)
        match dict_action["action"]:
            case Actions.CUT_CIRCUIT:
                penalty = self._cut(*dict_action["params"])
            case Actions.COMBINE_CIRCUIT:
                self._combine(*dict_action["params"])
            case Actions.MOVE_CIRCUIT:
                self._move(*dict_action["params"])
            case Actions.SWAP_CIRCUITS:
                self._swap(*dict_action["params"])
            case Actions.TERMINATE:
                terminated = True

        # Calculate the completion time and noise based on the updated schedule
        # Return the new schedule, completion time, noise, and whether the task is done
        self._action_space.update_actions(self._schedule)
        if is_feasible(self._schedule):
            self._action_space.enable_terminate()
        else:
            self._action_space.disable_terminate()
        self.action_space = spaces.flatten_space(self._action_space)
        if self.steps >= self.max_steps:
            truncated = True
        self.steps += 1

        obs = self._get_observation()
        reward = (
            self._calculate_reward(float(obs["makespan"]), float(obs["noise"]))
            * penalty
        )
        if terminated and not is_feasible(self._schedule):
            reward = -np.inf

        return (
            obs,
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _unflatten_action(self, sample: np.ndarray) -> dict[str, Any]:
        unflattened_sample = {}
        current_index = 0
        for key, space_i in self._action_space.items():
            space_i_shape = spaces.utils.flatdim(space_i)
            space_i_sample = sample[current_index : current_index + space_i_shape]
            if isinstance(space_i, spaces.MultiDiscrete):
                multi_sample = ()
                sub_index = 0
                for space_j in space_i.nvec:
                    multi_sample += (
                        space_i_sample[sub_index : sub_index + space_j].astype(np.int8),
                    )
                    current_index += space_j
                unflattened_sample[key] = space_i.sample(multi_sample)

            else:
                unflattened_sample[key] = space_i.sample(space_i_sample.astype(np.int8))
            current_index += space_i_shape
        return unflattened_sample

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        schedule = Schedule(
            [
                Machine(accelerator.qubits, str(accelerator.uuid), [], np.inf)
                for accelerator in self.accelerators
            ],
            np.inf,
        )
        for circuit in self.circuits:
            choice = np.random.choice(len(schedule.machines))
            schedule.machines[choice].buckets.append(Bucket([circuit]))

        obs = self._get_observation()
        info = self._get_info()
        return (obs, info)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None

    def _get_observation(self) -> dict[str, float]:
        # Return the current observation of the environment
        # Example observation: (makespan, noise)
        self._schedule = evaluate_solution(self._schedule, self.accelerators)
        return {
            "makespan": np.array([self._schedule.makespan]),
            "noise": np.array([0.0]),
        }

    def _get_info(self) -> dict[str, Any]:
        # Return the current information of the environment
        return {
            machine.id: [
                [(job.circuit.num_qubits, str(job.uuid)) for job in bucket.jobs]
                for bucket in machine.buckets
            ]
            for machine in self._schedule.machines
        }

    def _calculate_reward(self, completion_time: float, expected_noise: float) -> float:
        # Calculate the reward based on the completion time and expected noise
        return -completion_time + expected_noise

    def _cut(self, index: int, cut_index: int, *_) -> float:
        # Cut the circuit into two smaller circuits
        # remove the circuit from the machine and add the newly created circuits
        # adds to the end
        (machine_id, bucket_id, job_id) = _find_job(self._schedule, index)
        job = self._schedule.machines[machine_id].buckets[bucket_id].jobs.pop(job_id)
        if job.circuit is None:
            raise ValueError("Job has no circuit")
        if (
            job.circuit.num_qubits < 3
            or job.circuit.num_qubits - cut_index < 2
            or cut_index < 2
        ):
            return self.penalty
        new_jobs = _convert_to_jobs(
            [job.circuit],
            [[0] * cut_index + [1] * (job.circuit.num_qubits - cut_index)],
        )
        self._schedule.machines[machine_id].buckets[bucket_id].jobs += new_jobs
        return 1

    def _combine(self, index1: int, index2: int, *_) -> None:
        # Combine two circuits into a single larger circuit
        # remove the two circuits from the machine and add the larger circuit
        # adds to the bucket of the first circuit
        (machine_id1, bucket_id1, job_id1) = _find_job(self._schedule, index1)
        (machine_id2, bucket_id2, job_id2) = _find_job(self._schedule, index2)
        job_1 = (
            self._schedule.machines[machine_id1].buckets[bucket_id1].jobs.pop(job_id1)
        )
        job_2 = (
            self._schedule.machines[machine_id2].buckets[bucket_id2].jobs.pop(job_id2)
        )
        combined_circuit = assemble_job([job_1, job_2])

        self._schedule.machines[machine_id1].buckets[bucket_id1].jobs.append(
            combined_circuit
        )

    def _move(self, index1: int, _: int, move_to: int) -> None:
        # Move a circuit to a new bucket
        (machine_id, bucket_id, job_id) = _find_job(self._schedule, index1)
        (new_machine_id, new_bucket_id) = _find_bucket(self._schedule, move_to)
        job = self._schedule.machines[machine_id].buckets[bucket_id].jobs.pop(job_id)
        self._schedule.machines[new_machine_id].buckets[new_bucket_id].jobs.append(job)

    def _swap(self, index1: int, index2: int, *_) -> None:
        (machine_id1, bucket_id1, job_id1) = _find_job(self._schedule, index1)
        (machine_id2, bucket_id2, job_id2) = _find_job(self._schedule, index2)

        (
            self._schedule.machines[machine_id1].buckets[bucket_id1].jobs[job_id1],
            self._schedule.machines[machine_id2].buckets[bucket_id2].jobs[job_id2],
        ) = (
            self._schedule.machines[machine_id2].buckets[bucket_id2].jobs[job_id2],
            self._schedule.machines[machine_id1].buckets[bucket_id1].jobs[job_id1],
        )


def _find_job(schedule: Schedule, index: int) -> tuple[int, int, int]:
    count = 0
    for machine_id, machine in enumerate(schedule.machines):
        for bucket_id, bucket in enumerate(machine.buckets):
            for job_id, _ in enumerate(bucket.jobs):
                if count == index:
                    return machine_id, bucket_id, job_id
                count += 1
    raise ValueError("Index out of range")


def _find_bucket(schedule: Schedule, index: int) -> tuple[int, int]:
    count = 0
    for machine_id, machine in enumerate(schedule.machines):
        machine.buckets.append(Bucket([]))  # allow to create new bucket
        for bucket_id, _ in enumerate(machine.buckets):
            if count == index:
                return machine_id, bucket_id
            count += 1
        machine.buckets.pop()
    raise ValueError("Index out of range")