"""A scheduler for quantum circuits."""
from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from src.common import (
    CircuitJob,
    CombinedJob,
    jobs_from_experiment,
    job_from_circuit,
    ScheduledJob,
)
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


class Scheduler:
    """The scheduer is aware of the hardware and schedules circuits accordingly.
    At the moment, scheduling is done offline.
    New circuits can't be submitted while the scheduler is running.
    TODO:   - Raise a flag when an experiment is done (using its UUID)
            - Consider number of shots / time to run circuit
            - Consider 1 free qubit remaining when scheduling
            - Make a continuous run function / sumbit new circuits
            - Keep track of current schedule and update it
    """

    def __init__(self, accelerators: list[Accelerator]) -> None:
        self.jobs = []
        self.accelerator = AcceleratorGroup(accelerators)
        self.uuids = []

    def run_circuits(self, circuits: list[QuantumCircuit]) -> list[CombinedJob]:
        """Genreates a schedule and runs it.

        Args:
            circuits (list[QuantumCircuit]): Circuits to run.

        Returns:
            list[CombinedJob]: Jobs with inserted results.
        """
        jobs = self.generate_schedule(circuits)
        return self.accelerator.run_jobs(jobs)

    def generate_schedule(self, circuits: list[QuantumCircuit]) -> list[ScheduledJob]:
        """Generates an offlines schedule.

        First cuts the circuits which are too big to run on the biggest qpu.
        Circuits are cut to fit onto the biggest qpu first.
        Flattens the resulting experiment and creates individual jobs for each circuit.
        The jobs are scheduled using k-first fit bin packing.
        As soon as a bin is full, a new bin for each qpu is added.
        This is based on the assumption that all qpus take the same amount of time to run.


        Args:
            circuits (list[ScheduledJob]): A list of Jobs ready to run.
                The jobs are sorted by index (the timestep when to run)
                and have qpu information attached.
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
        """Schedule jobs onto qpus.

        Each qpu represents a bin.
        Since all jobs are asumet to take the same amount of time, the are associated
        with a timestep (index).
        k-first fit bin means we keep track of all bins that still have space left.
        Once a qpu is full, we add a new bin for each qpu at the next timestep.
        We can't run circuits with one qubit, scheduling doesn't take this into account.
        Args:
            jobs (list[CircuitJob]): The list of jobs to run.

        Returns:
            list[Bin]: All bins with at least one jobs.
        """
        # Use binpacking to combine circuits into qpu sized jobs
        # placeholder for propper scheduling
        # TODO set a flag when an experiment is done
        # TODO consider number of shots
        # Assumption: beens should be equally loaded and take same amoutn of time
        open_bins = [
            Bin(index=0, capacity=qpu.qubits, qpu=idx)
            for idx, qpu in enumerate(self.accelerator.accelerators)
        ]
        closed_bins = []
        index = 1
        for job in jobs:
            for obin in open_bins:
                # TODO consider 1 free qubit remaining
                if obin.capacity >= job.instance.num_qubits:
                    obin.jobs.append(job)
                    obin.capacity -= job.instance.num_qubits
                    if obin.capacity <= 1:
                        obin.full = True
                        closed_bins.append(obin)
                        open_bins.remove(obin)
                    break
            else:
                new_bins = [
                    Bin(index=index, capacity=qpu.qubits, qpu=idx)
                    for idx, qpu in enumerate(self.accelerator.accelerators)
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
        """Generates jobs from circuits.

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
        qpu_sizes = [acc.qubits for acc in self.accelerator.accelerators]
        for circuit_size in circuit_sizes:
            if circuit_size > self.accelerator.qubits:
                partition = qpu_sizes
                remaining_size = circuit_size - self.accelerator.qubits
                while remaining_size > self.accelerator.qubits:
                    partition += qpu_sizes
                    remaining_size -= self.accelerator.qubits
                if remaining_size == 1:
                    partition[-1] = partition[-1] - 1
                    partition.append(2)
                else:
                    partition.append(self._partition_big_to_small(remaining_size))
                partitions.append(partition)
            elif circuit_size > max(qpu_sizes):
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
        for qpu in sorted(
            self.accelerator.accelerators, key=lambda a: a.qubits, reverse=True
        ):
            take_qubits = min(size, qpu.qubits)
            if size - take_qubits == 1:
                # We can't have a partition of size 1
                # So in this case we take one qubit less to leave a partition of two
                take_qubits -= 1
            
            partition.append(take_qubits)
            size -= take_qubits
            if size == 0:
                break
        else:
            raise ValueError(f"Circuit is too big to fit onto the devices, {size} qubits left after partitioning.")
        return partition
