import argparse

# Import the problem description
import data.milp as milp

# Parse the command line arguments
parser = argparse.ArgumentParser(
    description="Print setup and process times as LaTeX table"
)
parser.add_argument(
    "machine",
    type=int,
    help="The machine to print the times for",
)
parser.add_argument(
    "mode",
    help="What times to print",
    choices=["setup", "process"],
)
args = parser.parse_args()


def print_row(row: list[str]):
    print(" & ".join(row) + " \\\\")


mode = args.mode
m = args.machine
machine = milp.machines[m]

# Print the header
header = [machine] + milp.jobs
print_row(header)

# Print the table
for i, job_i in enumerate(milp.jobs):
    row = [job_i]
    if mode == "setup":
        for j, job_j in enumerate(milp.jobs):
            if job_i == job_j:
                item = "-"
            else:
                s_ijk = milp.get_setup_time(i, j, m)
                item = str(s_ijk)
            row.append(f"{item:4}")
    elif mode == "process":
        p_ik = milp.get_process_time(i, m)
        row.append(f"{p_ik:4}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print_row(row)
