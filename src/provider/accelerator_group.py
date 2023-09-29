""""""
from itertools import zip_longest
from multiprocessing import Pool, current_process

from qiskit import QuantumCircuit

from src.common import CombinedJob, Experiment, ScheduledJob
from .accelerator import Accelerator


class AcceleratorGroup:
    """_summary_"""

    def __init__(self, accelerators: list[Accelerator]) -> None:
        self.accelerators = accelerators
        self._qpu_qubits = [acc.qubits for acc in accelerators]

    @property
    def qpus(self) -> list[int]:
        """_summary_

        Returns:
            list[int]: _description_
        """
        return self._qpu_qubits

    @property
    def qubits(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return sum(self._qpu_qubits)

    def run_and_get_counts(
        self, circuits: list[QuantumCircuit]
    ) -> list[dict[int, int]]:
        """_summary_

        Args:
            circuits (list[QuantumCircuit]): _description_

        Returns:
            list[dict[int, int]]: _description_
        """
        counts = []
        for circuit, accelerator in zip(circuits, self.accelerators):
            # TODO in parallel!
            counts.append(accelerator.run_and_get_counts(circuit))
        # TODO do some magic to figure out which counts belong to which circuit
        return counts

    def run_jobs(self, jobs: list[ScheduledJob]) -> list[CombinedJob]:
        """_summary_

        Args:
            jobs (list[ScheduledJob]): _description_

        Returns:
            list[ScheduledJob]: _description_
        """
        jobs_per_qpu = {
            qpu: [job for job in jobs if job.qpu == qpu]
            for qpu, _ in enumerate(self.qpus)
        }
        with Pool(processes=len(self.accelerators)) as pool:
            results = []
            for job in zip_longest(*jobs_per_qpu.values()):
                result = pool.apply_async(_run_job, [self.accelerators, job])
                results.append(result)
            results = [result.get() for result in results]
        results = [result for result in results if result is not None]
        return results

    def run_experiments(self, experiments: list[Experiment]) -> list[Experiment]:
        """_summary_

        Args:
            experiments (list[Experiment]): _description_

        Returns:
           list[Experiment]: _description_
        """
        with Pool(processes=len(self.accelerators)) as pool:
            results = []
            for experiment in experiments:
                result = pool.apply_async(_run_func, [self.accelerators, experiment])
                results.append(result)
            results = [result.get() for result in results]

        return results


def _run_func(accs: list[Accelerator], exp: Experiment) -> Experiment:
    pool_id = current_process()._identity[0] - 1  # TODO fix somehow
    try:
        exp.result_counts = [
            accs[pool_id].run_and_get_counts(circ) for circ in exp.circuits
        ]
    except Exception as e:
        print(e)
    return exp


def _run_job(
    accs: list[Accelerator], jobs: tuple[CombinedJob | None]
) -> CombinedJob | None:
    pool_id = current_process()._identity[0] - 1  # TODO fix somehow
    job = jobs[pool_id]
    if job is None:
        return None
    job = job.job
    try:
        job.result_counts = accs[pool_id].run_and_get_counts(job.instance)

    except Exception as e:
        print(e)
    return job
