"""A common interface for multiple accelerators."""
from itertools import zip_longest
from multiprocessing import Pool, current_process, Manager, Queue
from qiskit import QuantumCircuit

from src.common import CombinedJob, Experiment, ScheduledJob
from .accelerator import Accelerator


class AcceleratorGroup:
    """
    Provides a single entrypoint for multiple accelerators.
    """

    def __init__(self, accelerators: list[Accelerator]) -> None:
        self._accelerators = accelerators
        self._qpu_qubits = [acc.qubits for acc in accelerators]

    @property
    def qpus(self) -> list[int]:
        """Indicates the number of qubits per qpu.

        Returns:
            list[int]: The number of qubits per qpu.
        """
        return self._qpu_qubits

    @property
    def accelerators(self) -> list[Accelerator]:
        """Get the wrapped accelerators.

        Returns:
            list[Accelerator]: The internal accelerators.
        """
        return self._accelerators

    @property
    def qubits(self) -> int:
        """Tolal number of qubits.

        Returns:
            int: The total number of qubits.
        """
        return sum(self._qpu_qubits)

    def run_and_get_counts(
        self, circuits: list[QuantumCircuit]
    ) -> list[dict[int, int]]:
        """Simple run and get counts simultaneously for all accelerators.

        Only works if the number of circuits is equal to the number of accelerators.
        Args:
            circuits (list[QuantumCircuit]): The circuits to run.

        Returns:
            list[dict[int, int]]: A list of result counts, preserving order.
        """
        counts = []
        for circuit, accelerator in zip(circuits, self._accelerators):
            # TODO in parallel!
            counts.append(accelerator.run_and_get_counts(circuit))
        # TODO do some magic to figure out which counts belong to which circuit
        return counts

    def run_jobs(self, jobs: list[ScheduledJob]) -> list[CombinedJob]:
        """Runs a list of scheduled jobs on their respective accelerators.

        Creates a tuple of jobs to run at the same time onf different qpus.
        Jobs are run in parallel.
        Args:
            jobs (list[ScheduledJob]): The jobs to run.

        Returns:
            list[ScheduledJob]: The jobs with results inserted.
        """
        jobs_per_qpu = {
            qpu: [job for job in jobs if job.qpu == qpu]
            for qpu, _ in enumerate(self.accelerators)
        }  # Sort by qpu
        manager = Manager()
        idx_queue = manager.Queue()
        for idx in range(len(self._accelerators)):
            idx_queue.put(idx)
        with Pool(
            processes=len(self._accelerators),
            initializer=_init_accs,
            initargs=(idx_queue,),
        ) as pool:
            results = []
            for job in zip_longest(*jobs_per_qpu.values()):  # Run jobs in parallel
                result = pool.apply_async(_run_job, [self._accelerators, job])
                results.append(result)
            results = [result.get() for result in results]
        results = [result for result in results if result is not None]
        return results

    def run_experiments(self, experiments: list[Experiment]) -> list[Experiment]:
        """Runs all circuits belonging to a list of experiments.

        Just takes the circuits without any consideration.
        Circuits are run in parallel.

        Args:
            experiments (list[Experiment]): The list of expermimets to run.

        Returns:
           list[Experiment]: Experiment with results inserted.
        """
        manager = Manager()
        idx_queue = manager.Queue()
        for idx in range(len(self._accelerators)):
            idx_queue.put(idx)
        with Pool(
            processes=len(self._accelerators),
            initializer=_init_accs,
            initargs=(idx_queue,),
        ) as pool:
            results = []
            for experiment in experiments:
                result = pool.apply_async(_run_func, [self._accelerators, experiment])
                results.append(result)
            results = [result.get() for result in results]

        return results


def _init_accs(queue: Queue) -> None:
    idx = queue.get()
    current_process().name = str(idx)


def _run_func(accs: list[Accelerator], exp: Experiment) -> Experiment:
    """Wrapper to run Experiment on a single accelerator.

    Args:
        accs (list[Accelerator]): The internal accelerators.
        exp (Experiment): The experiment to run on this accelerator.

    Returns:
        Experiment: Experiment with results inserted.

    """
    pool_id = int(current_process().name)
    try:
        exp.result_counts = [
            accs[pool_id].run_and_get_counts(circ) for circ in exp.circuits
        ]
    except Exception as exc:
        # To make result.get() work deterministically
        print(exc)
    return exp


def _run_job(
    accs: list[Accelerator], jobs: tuple[ScheduledJob | None]
) -> CombinedJob | None:
    """Selects a job from a tuple of jobs and runs it on the respective accelerator.

    Args:
        accs (list[Accelerator]): the internal accelerators.
        jobs (tuple[ScheduledJob  |  None]): The jobs which are run in parallel.

    Returns:
        CombinedJob | None: Returns the job with results inserted or None if no job was submited.
    """
    pool_id = int(current_process().name)
    if pool_id > len(accs):
        return None
    job = jobs[pool_id]
    if job is None:
        return None
    run_job = job.job
    try:
        run_job.result_counts = accs[pool_id].run_and_get_counts(run_job.circuit, run_job.n_shots)
    except Exception as exc:
        print(exc)
    return run_job
