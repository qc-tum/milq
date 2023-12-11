"""_summary_"""

from src.circuits import create_ghz
from src.common import IBMQBackend, job_from_circuit
from src.provider import Accelerator
from src.provider.generate_schedule import generate_simple_schedule


def test_generate_simple_schedule() -> None:
    """_summary_"""
    accelerators = [
        Accelerator(IBMQBackend.BELEM, shot_time=1, reconfiguration_time=1),
        Accelerator(IBMQBackend.QUITO, shot_time=1, reconfiguration_time=1),
    ]
    jobs = [job_from_circuit(create_ghz(i)) for i in range(2, 6)]
    schedule = generate_simple_schedule(jobs, accelerators, t_max=20)
    assert len(schedule) <= 3
