"""_summary_"""
from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from src.common import CircuitJob, CombinedJob, jobs_from_experiment, job_from_circuit
from src.tools import cut_circuit, assemble_job

from .accelerator import Accelerator
from .accelerator_group import AcceleratorGroup


@dataclass
class Bin:
    """Helper to keep track of binning problem."""

    capacity: int = 0
    full: bool = False
    index: int = -1
    jobs: list[CircuitJob] = field(default_factory=list)
    qpu: int = -1


@dataclass
class ScheduledJob:
    """Data class for scheduled job."""

    job: CombinedJob  # Probably don't need CircuitJob
    qpu: int


# Some thoughts
# Preprocess such that every circuit is smaller than the max QPU size
# And each (sub)circuit is assigned to a hardware
# 1. cut everything thats to large -> Experiments
# 2. Put everything that's small into experiment
# 3. combine circuits that fit into small device
class Scheduler:
    """_summary_"""

    def __init__(self, accelerators: list[Accelerator]) -> None:
        self.jobs = []
        self.accelerator = AcceleratorGroup(accelerators)
        self.uuids = []

    def generate_schedule(self, circuits: list[QuantumCircuit]) -> list[ScheduledJob]:
        """_summary_

        Args:
            circuits (list[ScheduledJob]): _description_
        """
        jobs = sorted(
            self._convert_to_jobs(circuits),
            key=lambda x: x.instance.num_qubits,
            reverse=True,
        )
        bins = self._binpacking_to_qpus(jobs)
        combined_jobs = []
        for _bin in sorted(bins, key=lambda x: x.index):
            combined_jobs.append(
                ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu)
            )
        return combined_jobs

    def _binpacking_to_qpus(self, jobs: list[CircuitJob]) -> list[Bin]:
        # Use binpacking to combine circuits into qpu sized jobs
        # placeholder for propper scheduling
        # TODO set a flag when an experiment is done
        # TODO consider number of shots
        # Assumption: beens should be equally loaded and take same amoutn of time
        open_bins = [
            Bin(index=0, capacity=qpu, qpu=idx)
            for idx, qpu in enumerate(self.accelerator.qpus)
        ]
        closed_bins = []
        index = 1
        for job in jobs:
            for obin in open_bins:
                # TODO consider 1 free qubit remaining
                if obin.capacity >= job.instance.num_qubits:
                    obin.jobs.append(job)
                    obin.capacity -= job.instance.num_qubits
                    if obin.capacity == 0:
                        obin.full = True
                        closed_bins.append(obin)
                        open_bins.remove(obin)
                    break
            else:
                new_bins = [
                    Bin(index=index, capacity=qpu, qpu=idx)
                    for idx, qpu in enumerate(self.accelerator.qpus)
                ]
                index += 1
                for nbin in new_bins:
                    # TODO consider 1 free qubit remaining
                    if nbin.capacity >= job.instance.num_qubits:
                        nbin.jobs.append(job)
                        nbin.capacity -= job.instance.num_qubits
                        if nbin.capacity == 0:
                            nbin.full = True
                            closed_bins.append(nbin)
                            new_bins.remove(nbin)
                        break
                open_bins += new_bins
        for obin in open_bins:
            if len(obin.jobs) > 0:
                closed_bins.append(obin)
        return closed_bins

    def _convert_to_jobs(self, circuits: list[QuantumCircuit]) -> list[CircuitJob]:
        """Generater jobs from circuits.

        Small circuits are converted to jobs directly.
        Big circuits are cut into experiments and turned into jobs.

        Args:
            circuits (list[QuantumCircuit]): List of circuits to convert.

        Returns:
            list[CircuitJob]: Circuit jobs.
        """
        partitions = self._generate_partitions(
            [circuit.num_qubits for circuit in circuits]
        )
        jobs = []
        for idx, circuit in enumerate(circuits):
            if len(partitions[idx]) > 1:
                experiments, uuid = cut_circuit(circuit, partitions[idx])
                self.uuids.append(uuid)
                jobs += [
                    job
                    for experiment in experiments
                    for job in jobs_from_experiment(experiment)
                ]
            else:
                # assumption for now dont cut to any to smaller
                circuit = job_from_circuit(circuit)
                jobs.append(circuit)
                self.uuids.append(circuit.uuid)
        return jobs

    def _generate_partitions(self, circuit_sizes: list[int]) -> list[list[int]]:
        """Generates partitions for the given circuit sizes.
        Order of the partitions is the same as the order of the circuits.
        First circuits are cut until they fit onto the full device, then the remainder
        is cut into the biggest QPUs first.

        Args:
            circuit_sizes (list[int]): Sizes of the invividual circuits.

        Returns:
            list[list[int]]: List of partitions for each circuit.
        """
        # For now we partition everything from biggest qpu to smallest
        # TODO: This is a very naive approach, we should do something smarter
        # for example: look at existing partitions and try to fill up the remaining space
        partitions = []
        for circuit_size in circuit_sizes:
            if circuit_size > self.accelerator.qubits:
                partition = self.accelerator.qpus
                remaining_size = circuit_size - self.accelerator.qubits
                while remaining_size > self.accelerator.qubits:
                    partition += self.accelerator.qpus
                    remaining_size -= self.accelerator.qubits
                if remaining_size == 1:
                    partition[-1] = partition[-1] - 1
                    partition.append(2)
                else:
                    partition.append(self._partition_big_to_small(remaining_size))
                partitions.append(partition)
            elif circuit_size > max(self.accelerator.qpus):
                partition = self._partition_big_to_small(circuit_size)
                partitions.append(partition)
            else:
                partitions.append([circuit_size])
        return partitions

    def _partition_big_to_small(self, size: int) -> list[int]:
        """Partitions a circuit into the biggest QPUs first.

        Args:
            size (int): The size of the (remaining) circuit to partition.

        Returns:
            list[int]: The partition sizes of the circuit.
        """
        partition = []
        for qpu in sorted(self.accelerator.qpus, reverse=True):
            if size >= qpu:
                partition.append(qpu)
                size -= qpu
            else:
                partition.append(size)
                break
        if size == 1:
            partition[-2] = partition[-2] - 1
            partition[-1] = 2
        return partition
