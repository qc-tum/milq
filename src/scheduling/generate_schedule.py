"""Wrapper for schedule generation."""
from uuid import uuid4

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator

from .bin_schedule import (
    generate_bin_info_schedule,
    generate_bin_executable_schedule,
)
from .calculate_makespan import calculate_makespan, calculate_bin_makespan
from .extract_schedule import extract_info_schedule, extract_executable_schedule
from .setup_lp import set_up_base_lp, set_up_extended_lp, set_up_simple_lp
from .solve_lp import solve_lp
from .types import (
    ExecutableProblem,
    InfoProblem,
    JobResultInfo,
    LPInstance,
    PTimes,
    SchedulerType,
    STimes,
)


def generate_schedule(
    problem: InfoProblem | ExecutableProblem,
    schedule_type: SchedulerType,
) -> tuple[float, list[JobResultInfo], LPInstance | None] | list[ScheduledJob]:
    """Generates the schedule for the given problem and schedule type.

    Baseline: Generates a schedule using binpacking.
    Else generates the schedule using MILP  and then calculates the makespan
    by executing the schedule with the correct p_ij and s_ij values.
    Args:
        problem (InfoProblem | ExecutableProblem ): The full problem definition.
        schedule_type (SchedulerType): The type of schedule to use.

    Returns:
        list[ScheduledJob]: List of ScheduledJobs. |
        tuple[float, list[JobResultInfo]]: The makespan and the list of jobs with their
        assigned machine and start and completion times.

    Raises:
        NotImplementedError: Unsupported types.
    """
    if isinstance(problem, InfoProblem):
        return _generate_schedule_info(problem, schedule_type)
    if isinstance(problem, ExecutableProblem):
        return _generate_schedule_exec(problem, schedule_type)
    raise NotImplementedError("Unsupported type")


def _generate_schedule_info(
    problem: InfoProblem,
    schedule_type: SchedulerType,
) -> tuple[float, list[JobResultInfo], LPInstance | None]:
    """Generates the schedule for the given problem and schedule type.

    Calculates the true makespan by 'executing' the schedlue.
    Args:
        problem (InfoProblem): The full problem definition.
        schedule_type (SchedulerType): The type of schedule to use.

    Returns:
        tuple[float, list[JobResultInfo]]: The makespan and the list of jobs with their
            assigned machine and start and completion times.
    """
    if schedule_type == SchedulerType.BASELINE:
        jobs = generate_bin_info_schedule(problem.base_jobs, problem.accelerators)
        makespan = calculate_bin_makespan(
            jobs, problem.process_times, problem.setup_times, problem.accelerators
        )
        return makespan, jobs, None

    lp_instance = set_up_base_lp(
        problem.base_jobs, problem.accelerators, problem.big_m, problem.timesteps
    )

    if schedule_type == SchedulerType.EXTENDED:
        lp_instance = set_up_extended_lp(
            lp_instance=lp_instance,
            process_times=problem.process_times,
            setup_times=problem.setup_times,
        )
    else:
        lp_instance = set_up_simple_lp(
            lp_instance=lp_instance,
            process_times=problem.process_times,
            setup_times=problem.setup_times,
        )

    lp_instance = solve_lp(lp_instance)
    _, jobs = extract_info_schedule(lp_instance)
    makespan = calculate_makespan(
        lp_instance, jobs, problem.process_times, problem.setup_times
    )

    return makespan, jobs, lp_instance


def _generate_schedule_exec(
    problem: ExecutableProblem,
    schedule_type: SchedulerType,
) -> list[ScheduledJob]:
    """Generates the schedule for the given problem and schedule type.

    Process and setup times are calculated on the fly.
    The jobs are returned in a format that can be executed by AcceleratorGroup.
    Args:
        problem (ExecutableProblem): The full problem definition.
        schedule_type (SchedulerType): The type of schedule to use.

    Returns:
        list[ScheduledJob]: List of ScheduledJobs.
    """
    if schedule_type == SchedulerType.BASELINE:
        return generate_bin_executable_schedule(problem.base_jobs, problem.accelerators)

    lp_instance = set_up_base_lp(
        problem.base_jobs, problem.accelerators, problem.big_m, problem.timesteps
    )
    process_times = _get_processing_times(problem.base_jobs, problem.accelerators)
    setup_times = _get_setup_times(problem.base_jobs, problem.accelerators)
    if schedule_type == SchedulerType.EXTENDED:
        lp_instance = set_up_extended_lp(
            lp_instance=lp_instance,
            process_times=process_times,
            setup_times=setup_times,
        )
    else:
        lp_instance = set_up_simple_lp(
            lp_instance=lp_instance,
            process_times=process_times,
            setup_times=setup_times,
        )

    lp_instance = solve_lp(lp_instance)
    return extract_executable_schedule(
        lp_instance, problem.base_jobs, problem.accelerators
    )


def _get_setup_times(
    base_jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> STimes:
    job_0 = CircuitJob(
        coefficient=None,
        cregs=0,
        index=0,
        circuit=None,
        n_shots=0,
        observable="",
        partition_label="",
        result_counts=None,
        uuid=uuid4(),
    )
    return [
        [
            [
                50.0
                if job_i.circuit is None or job_j.circuit is None
                else qpu.compute_setup_time(job_i.circuit, job_j.circuit)
                for qpu in accelerators
            ]
            for job_i in [job_0] + base_jobs
        ]
        for job_j in [job_0] + base_jobs
    ]


def _get_processing_times(
    base_jobs: list[CircuitJob],
    accelerators: list[Accelerator],
) -> PTimes:
    return [[0.0 for _ in accelerators]] + [
        [qpu.compute_processing_time(job.circuit) for qpu in accelerators]
        for job in base_jobs
        if job.circuit is not None
    ]
