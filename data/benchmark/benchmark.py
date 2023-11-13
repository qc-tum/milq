"""Generates the benchmark data."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Collection
import json

from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
import numpy as np
import pulp

np.random.seed(42)

# Define the maximum circuit size
# MAX_SIZE = 25
NUM_BATCHES = 10
CIRCUITS_PER_BATCH = 5
SETTINGS = [
    {"A": 5, "B": 5},
    {"A": 5, "B": 6, "C": 20},
]
T_MAX = 2**6


@dataclass
class Bin:
    """Helper to keep track of binning problem."""

    capacity: int = 0
    full: bool = False
    index: int = -1
    jobs: list[QuantumCircuit] = field(default_factory=list)
    qpu: int = -1


@dataclass
class LPInstance:
    """Helper to keep track of LP problem."""

    problem: pulp.LpProblem
    jobs: list[str]
    machines: list[str]
    x_ik: dict[str, dict[str, pulp.LpVariable]]
    z_ikt: dict[str, dict[str, dict[int, pulp.LpVariable]]]
    c_j: dict[str, pulp.LpVariable]
    s_j: dict[str, pulp.LpVariable]


@dataclass
class JobResultInfo:
    """Helper to keep track of job results."""

    name: str
    machine: str = ""
    start_time: float = -1.0
    completion_time: float = -1.0


@dataclass
class Result:
    makespan: float
    jobs: list[JobResultInfo]

def generate_batch(max_size: int, circuits_per_batch: int) -> list[QuantumCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_size + 1)
        circuit = get_benchmark(benchmark_name="random", level=0, circuit_size=size)
        batch.append(circuit)

    return batch


def run_experiments(
    circuits_per_batch: int, settings: list[dict[str, int]]
) -> list[dict[str, Collection[Collection[str]]]]:
    results = []
    for setting in settings:
        max_size = max(setting.values())
        benchmarks = [
            generate_batch(max_size, circuits_per_batch) for _ in range(NUM_BATCHES)
        ]
        benchmark_results = []
        for benchmark in benchmarks:
            # TODO: timing
            # TODO: set up processing times and set up times
            # TODO: timesteps
            lp_instance = _set_up_base_lp(
                benchmark, setting, big_m=1000, timesteps=list(range(2**6))
            )
            p_times = pulp.makeDict(
                [lp_instance.jobs[1:], lp_instance.machines],
                _get_processing_times(benchmark, setting),
                0,
            )
            s_times = pulp.makeDict(
                [lp_instance.jobs[1:], lp_instance.machines],
                _get_simple_setup_times(benchmark, setting),
                0,
            )
            result = {}
            # makespan, jobs = generate_baseline_schedule(benchmark, setting)
            result["baseline"] = Result(makespan, jobs)
            makespan, jobs = generate_simple_schedule(lp_instance, p_times, s_times)
            result["simple"] = Result(_calclulate_makespan_from_simple(jobs, p_times, s_times), jobs)
            makespan, jobs = generate_extended_schedule(lp_instance, p_times,s_times)
            result["extended"] = Result(makespan, jobs)
            benchmark_results.append(result)

            results.append({"setting": setting, "benchmarks": benchmark_results})
    return results


def _set_up_base_lp(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    big_m: int,
    timesteps: list[int],
) -> LPInstance:
    # Set up input params
    jobs = ["0"] + [str(idx + 1) for idx, _ in enumerate(base_jobs)]
    job_capacities = {str(idx + 1): job.num_qubits for idx, job in enumerate(base_jobs)}
    job_capacities["0"] = 0
    machines = [qpu for qpu in accelerators.keys()]
    machine_capacities = {qpu: qubits for qpu, qubits in accelerators.items()}

    # set up problem variables
    x_ik = pulp.LpVariable.dicts("x_ik", (jobs, machines), cat="Binary")
    z_ikt = pulp.LpVariable.dicts("z_ikt", (jobs, machines, timesteps), cat="Binary")

    c_j = pulp.LpVariable.dicts("c_j", (jobs), 0, cat="Continuous")
    s_j = pulp.LpVariable.dicts("s_j", (jobs), 0, cat="Continuous")
    c_max = pulp.LpVariable("makespan", 0, cat="Continuous")

    problem = pulp.LpProblem("Scheduling", pulp.LpMinimize)
    # set up problem constraints
    problem += pulp.lpSum(c_max)  # (obj)
    problem += c_j["0"] == 0  # (8)
    for job in jobs[1:]:
        problem += c_j[job] <= c_max  # (1)
        problem += pulp.lpSum(x_ik[job][machine] for machine in machines) == 1  # (3)
        problem += c_j[job] - s_j[job] + 1 == pulp.lpSum(  # (11)
            z_ikt[job][machine][timestep]
            for timestep in timesteps
            for machine in machines
        )
        for machine in machines:
            problem += (  # (12)
                pulp.lpSum(z_ikt[job][machine][timestep] for timestep in timesteps)
                <= x_ik[job][machine] * big_m
            )

        for timestep in timesteps:
            problem += (  # (13)
                pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines)
                * timestep
                <= c_j[job]
            )
            problem += s_j[job] <= pulp.lpSum(  # (14)
                z_ikt[job][machine][timestep] for machine in machines
            ) * timestep + big_m * (
                1 - pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines)
            )
    for timestep in timesteps:
        for machine in machines:
            problem += (  # (15)
                pulp.lpSum(
                    z_ikt[job][machine][timestep] * job_capacities[job]
                    for job in jobs[1:]
                )
                <= machine_capacities[machine]
            )
    return LPInstance(
        problem=problem,
        jobs=jobs,
        machines=machines,
        x_ik=x_ik,
        z_ikt=z_ikt,
        c_j=c_j,
        s_j=s_j,
    )


def _solve_lp(lp_instance: LPInstance) -> tuple[float, list[JobResultInfo]]:
    solver_list = pulp.listSolvers(onlyAvailable=True)
    gurobi = "GUROBI_CMD"
    if gurobi in solver_list:
        solver = pulp.getSolver(gurobi)
        lp_instance.problem.solve(solver)
    else:
        lp_instance.problem.solve()
    return _generate_results(lp_instance)


def _generate_results(lp_instance: LPInstance) -> tuple[float, list[JobResultInfo]]:
    assigned_jobs = {job: JobResultInfo(name=job) for job in lp_instance.jobs}
    for var in lp_instance.problem.variables():
        if var.name.startswith("x_") and var.varValue > 0.0:
            name = var.name.split("_")[2:]
            assigned_jobs["-".join(name[:5])].machine = "-".join(name[-5:])
        elif var.name.startswith("s_"):
            name = var.name.split("_")[2:]
            assigned_jobs[name].start_time = float(var.varValue)
        elif var.name.startswith("c_"):
            name = var.name.split("_")[2:]
            assigned_jobs[name].completion_time = float(var.varValue)
    del assigned_jobs["0"]
    return lp_instance.problem.objective.value(), list(assigned_jobs.values())


def generate_baseline_schedule(
    jobs: list[QuantumCircuit], accelerators: dict[str, int], **kwargs
) -> tuple[float, list[JobResultInfo]]:
    def find_fitting_bin(job: QuantumCircuit, bins: list[Bin]) -> int | None:
        for idx, b in enumerate(bins):
            if b.capacity >= job.num_qubits:
                return idx
        return None

    open_bins = [
        Bin(index=0, capacity=qpu, qpu=idx)
        for idx, qpu in enumerate(accelerators.values())
    ]
    closed_bins = []
    index = 1
    for job in jobs:
        if job.instance is None:
            continue
        # Find the index of a fitting bin
        bin_idx = find_fitting_bin(job, open_bins)

        if bin_idx is None:
            # Open new bins
            new_bins = [
                Bin(index=index, capacity=qpu, qpu=idx)
                for idx, qpu in enumerate(accelerators.values())
            ]
            index += 1

            # Search for a fitting bin among the new ones
            bin_idx = find_fitting_bin(job, new_bins)
            assert bin_idx is not None, "Job doesn't fit onto any qpu"
            bin_idx += len(open_bins)
            open_bins += new_bins

        # Add job to selected bin
        selected_bin = open_bins[bin_idx]
        selected_bin.jobs.append(job)
        selected_bin.capacity -= job.num_qubits

        # Close bin if full
        if selected_bin.capacity == 0:
            selected_bin.full = True
            closed_bins.append(selected_bin)
            del open_bins[bin_idx]

    # Close all open bins
    for obin in open_bins:
        if len(obin.jobs) > 0:
            closed_bins.append(obin)

    # Build combined jobs from bins
    combined_jobs: list[JobResultInfo] = []
    # TODO: calclulate makespan and schedule
    # for _bin in sorted(closed_bins, key=lambda x: x.index):
    #     combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
    return 0, combined_jobs


def generate_simple_schedule(
    lp_instance: LPInstance,
    p_times: defaultdict[str, defaultdict[str, float]],
    s_times: defaultdict[str, defaultdict[str, float]],
    big_m: int = 1000,

) -> tuple[float, list[JobResultInfo]]:
    # lp_instance = _set_up_base_lp(jobs, accelerators, big_m, list(range(t_max)))
    # # (4) - (7), (9)
    # p_times = pulp.makeDict(
    #     [lp_instance.jobs[1:], lp_instance.machines],
    #     _get_processing_times(jobs, accelerators),
    #     0,
    # )
    # s_times = pulp.makeDict(
    #     [lp_instance.jobs[1:], lp_instance.machines],
    #     _get_simple_setup_times(jobs, accelerators),
    #     0,
    # )
    y_ijk = pulp.LpVariable.dicts(
        "y_ijk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )

    for job in lp_instance.jobs[1:]:
        lp_instance.problem += (  # (4)
            pulp.lpSum(
                y_ijk[job_j][job][machine]
                for machine in lp_instance.machines
                for job_j in lp_instance.jobs
            )
            >= 1  # each job has a predecessor
        )
        lp_instance.problem += lp_instance.c_j[job] >= lp_instance.s_j[  # (7)
            job
        ] + pulp.lpSum(
            lp_instance.x_ik[job][machine]
            * (p_times[job][machine] + s_times[job][machine])
            for machine in lp_instance.machines
        )
        for machine in lp_instance.machines:
            lp_instance.problem += (  # predecessor (6)
                lp_instance.x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job_j][job][machine] for job_j in lp_instance.jobs)
                / big_m
            )
            lp_instance.problem += (  # successor
                lp_instance.x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job][job_j][machine] for job_j in lp_instance.jobs)
                / big_m
            )
            lp_instance.problem += (  # (5)
                lp_instance.z_ikt[job][machine][0] == y_ijk["0"][job][machine]
            )
        for job_j in lp_instance.jobs:
            lp_instance.problem += (
                lp_instance.c_j[job_j]
                + (
                    pulp.lpSum(
                        y_ijk[job_j][job][machine] for machine in lp_instance.machines
                    )
                    - 1
                )
                * big_m
                <= lp_instance.s_j[job]
            )

    return _solve_lp(lp_instance)


def generate_extended_schedule(
    lp_instance: LPInstance,
    p_times: defaultdict[str, defaultdict[str, float]],
    s_times: defaultdict[str, defaultdict[str, float]],
    big_m: int = 1000,
    **kwargs,
) -> tuple[float, list[JobResultInfo]]:
    # lp_instance = _set_up_base_lp(jobs, accelerators, big_m, list(range(t_max)))

    # # additional parameters
    # p_times = pulp.makeDict(
    #     [lp_instance.jobs, lp_instance.machines],
    #     _get_processing_times(jobs, accelerators),
    #     0,
    # )
    # s_times = pulp.makeDict(
    #     [lp_instance.jobs, lp_instance.jobs, lp_instance.machines],
    #     _get_setup_times(jobs, accelerators, kwargs.get("default_value", 50)),
    #     0,
    # )

    # decision variables
    y_ijk = pulp.LpVariable.dicts(
        "y_ijk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )
    a_ij = pulp.LpVariable.dicts(
        "a_ij", (lp_instance.jobs, lp_instance.jobs), cat="Binary"
    )  # a: Job i ends before job j starts
    b_ij = pulp.LpVariable.dicts(
        "b_ij", (lp_instance.jobs, lp_instance.jobs), cat="Binary"
    )  # b: Job i ends before job j ends
    d_ijk = pulp.LpVariable.dicts(
        "d_ijk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )  # d: Job i and  j run on the same machine
    e_ijlk = pulp.LpVariable.dicts(
        "e_ijlk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )

    for job in lp_instance.jobs[1:]:
        lp_instance.problem += (  # (4)
            pulp.lpSum(
                y_ijk[job_j][job][machine]
                for machine in lp_instance.machines
                for job_j in lp_instance.jobs
            )
            >= 1  # each job has a predecessor
        )
        lp_instance.problem += lp_instance.c_j[job] >= lp_instance.s_j[  # (7)
            job
        ] + pulp.lpSum(
            lp_instance.x_ik[job][machine] * p_times[job][machine]
            for machine in lp_instance.machines
        ) + pulp.lpSum(
            y_ijk[job_j][job][machine] * s_times[job_j][job][machine]
            for machine in lp_instance.machines
            for job_j in lp_instance.jobs
        )
        for machine in lp_instance.machines:
            lp_instance.problem += (  # predecessor (6)
                lp_instance.x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job_j][job][machine] for job_j in lp_instance.jobs)
                / big_m
            )
            lp_instance.problem += (  # successor
                lp_instance.x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job][job_j][machine] for job_j in lp_instance.jobs)
                / big_m
            )
            lp_instance.problem += (  # (5)
                lp_instance.z_ikt[job][machine][0] == y_ijk["0"][job][machine]
            )
        for job_j in lp_instance.jobs:
            lp_instance.problem += (
                lp_instance.c_j[job_j]
                + (
                    pulp.lpSum(
                        y_ijk[job_j][job][machine] for machine in lp_instance.machines
                    )
                    - 1
                )
                * big_m
                <= lp_instance.s_j[job]
            )

    # Extended constraints
    for job in lp_instance.jobs[1:]:
        for job_j in lp_instance.jobs[1:]:
            if job == job_j:
                lp_instance.problem += a_ij[job][job_j] == 0
                lp_instance.problem += b_ij[job][job_j] == 0
                continue
            lp_instance.problem += (
                a_ij[job][job_j]
                >= (lp_instance.s_j[job_j] - lp_instance.c_j[job]) / big_m
            )
            lp_instance.problem += (
                b_ij[job][job_j]
                >= (lp_instance.c_j[job_j] - lp_instance.c_j[job]) / big_m
            )
            for machine in lp_instance.machines:
                lp_instance.problem += (
                    d_ijk[job][job_j][machine]
                    >= lp_instance.x_ik[job][machine]
                    + lp_instance.x_ik[job_j][machine]
                    - 1
                )
                for job_l in lp_instance.jobs[1:]:
                    lp_instance.problem += (
                        e_ijlk[job][job_j][job_l][machine]
                        >= b_ij[job][job_l]
                        + a_ij[job_l][job_j]
                        + d_ijk[job][job_j][machine]
                        + d_ijk[job][job_l][machine]
                        - 3
                    )

    for job in lp_instance.jobs[1:]:
        for job_j in lp_instance.jobs[1:]:
            for machine in lp_instance.machines:
                lp_instance.problem += (
                    y_ijk[job][job_j][machine]
                    >= a_ij[job][job_j]
                    + (
                        pulp.lpSum(
                            e_ijlk[job][job_j][job_l][machine]
                            for job_l in lp_instance.jobs[1:]
                        )
                        / big_m
                    )
                    + d_ijk[job][job_j][machine]
                    - 2
                )
    return _solve_lp(lp_instance)


def _get_processing_times(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> list[list[float]]:
    # return [
    #     [qpu.compute_processing_time(job.instance) for qpu in accelerators]
    #     for job in base_jobs
    #     if job.instance is not None
    # ]
    # TODO
    return []


def _get_setup_times(
    base_jobs: list[QuantumCircuit], accelerators: dict[str, int], default_value: int
) -> list[list[list[float]]]:
    # return [
    #     [
    #         [
    #             default_value  # BIG!
    #             if job_i == job_j
    #             else qpu.compute_setup_time(job_i.instance, job_j.instance)
    #             for qpu in accelerators
    #         ]
    #         for job_i in base_jobs
    #         if job_i.instance is not None
    #     ]
    #     for job_j in base_jobs
    #     if job_j.instance is not None
    # ]
    # TODO
    return []


def _get_simple_setup_times(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> list[list[float]]:
    # return [
    #     [
    #         qpu.compute_setup_time(job_i.instance, circuit_to=None)
    #         for qpu in accelerators
    #     ]
    #     for job_i in base_jobs
    #     if job_i.instance is not None
    # ]
    # TODO
    return []


def _calclulate_makespan_from_simple(jobs: list[JobResultInfo],
    p_times: defaultdict[str, defaultdict[str, float]],
    s_times: defaultdict[str, defaultdict[str, float]],) -> float:
    return 0.0

def _calculate_result_from_baseline(jobs: list[JobResultInfo],
    p_times: defaultdict[str, defaultdict[str, float]],
    s_times: defaultdict[str, defaultdict[str, float]],) -> Result:
    return Result(0.0, [])
if __name__ == "__main__":
    experiment_results = run_experiments(CIRCUITS_PER_BATCH, SETTINGS)
    with open("benchmark_results.json", "w+", encoding="uft-8") as f:
        json.dump(experiment_results, f)

    # TODO: Visualize results
