"""_summary_"""

from src.circuits import create_ghz
from src.common import IBMQBackend, job_from_circuit
from src.provider import Accelerator
from src.scheduling import generate_schedule, SchedulerType, ExecutableProblem


def test_generate_schedule() -> None:
    """_summary_"""
    accelerators = [
        Accelerator(IBMQBackend.BELEM, shot_time=1, reconfiguration_time=1),
        Accelerator(IBMQBackend.QUITO, shot_time=1, reconfiguration_time=1),
    ]
    jobs = [job_from_circuit(create_ghz(i)) for i in range(2, 6)]
    problem = ExecutableProblem(jobs, accelerators, big_m=100, timesteps=20)

    schedule = generate_schedule(problem, SchedulerType.SIMPLE)
    assert isinstance(schedule, list)
    assert len(schedule) <= 4
