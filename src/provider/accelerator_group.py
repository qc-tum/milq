""""""
from multiprocessing import Pool, current_process

from qiskit import QuantumCircuit

from src.common import Experiment
from .accelerator import Accelerator


class AcceleratorGroup:
    """_summary_"""

    def __init__(self, accelerators: list[Accelerator]) -> None:
        self.accelerators = accelerators
        self._qubits = sum(acc.qubits for acc in accelerators)

    @property
    def qubits(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self._qubits

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
