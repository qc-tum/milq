"""Modlue for the example problem."""
import json

from qiskit import QuantumCircuit
import numpy as np

from src.scheduling import InfoProblem, generate_schedule, SchedulerType

np.random.seed(42)


def _calculate_exmaple_process_times(job_i, machine_k) -> float:
    if job_i == 0:
        return 0
    return job_i + np.random.randint(-2, 3) + machine_k


def _calculate_example_setup_times(job_i, job_j_, machine_k) -> float:
    if job_j_ == 0:
        return 0
    return (job_i + job_j_) // 2 + np.random.randint(-2, 3) + machine_k


def _generate_problem(big_m: int, timesteps: int) -> tuple[InfoProblem, dict[str, int]]:
    # Inputs
    jobs = ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I"]
    job_capacities = {
        "0": 0,  # dummy job
        "A": 5,
        "B": 5,
        "C": 5,
        "D": 5,
        "E": 3,
        "F": 2,
        "G": 2,
        "H": 2,
        "I": 2,
    }
    machines = ["QUITO", "BELEM"]
    machine_capacities = {"QUITO": 5, "BELEM": 5}

    processing_times = [
        [
            _calculate_exmaple_process_times(
                job_capacities[job], machine_capacities[machine]
            )
            for machine in machines
        ]
        for job in jobs
    ]
    setup_times = [
        [
            [
                50  # BIG!
                if job_i in [job_j, "0"]
                else _calculate_example_setup_times(
                    job_capacities[job_i],
                    job_capacities[job_j],
                    machine_capacities[machine],
                )
                for machine in machines
            ]
            for job_i in jobs
        ]
        for job_j in jobs
    ]
    del job_capacities["0"]
    return (
        InfoProblem(
            base_jobs=[QuantumCircuit(cap) for cap in job_capacities.values()],
            accelerators=machine_capacities,
            big_m=big_m,
            timesteps=timesteps,
            process_times=processing_times,
            setup_times=setup_times,
        ),
        job_capacities,
    )


def example_problem(big_m: int, timesteps: int, filename: str = "scheduling"):
    """Runs the example problem and saves the LP file and JSON file.
    TODO should also run the solution explorer and produce the output pdf.

    Args:
        big_m (int): LP metavariable.
        timesteps (int): LP metavariable.
        filename (str, optional): Filename for .lp, .json and .pdf. Defaults to "scheduling".
    """
    _problem, job_capacities = _generate_problem(big_m, timesteps)
    _, _, lp_instance = generate_schedule(_problem, SchedulerType.SIMPLE)
    lp_instance.problem.writeLP(f"{filename}.lp")

    with open(f"{filename}.json", "w+", encoding="utf-8") as f:
        json.dump(
            {
                "params": {
                    "jobs": list(job_capacities.keys()),
                    "machines": list(_problem.accelerators.keys()),
                    "job_capcities": job_capacities,
                    "machine_capacities": _problem.accelerators,
                    "timesteps": timesteps,
                    "processing_times": _problem.process_times,
                    "setup_times": _problem.setup_times,
                },
                "variables": {
                    var.name: var.varValue
                    for var in lp_instance.problem.variables()
                    if var.name.startswith(("c_j", "s_j", "x_ik_", "z_ikt_"))
                },
            },
            f,
            indent=4,
        )
