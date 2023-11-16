import argparse

# Fix relative imports
import sys
sys.path.append(__file__)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker
import pandas as pd

# Import the problem description
import milp

# Parse the command line arguments
parser = argparse.ArgumentParser(
    description="Visualize a solution to the scheduling problem"
)
parser.add_argument(
    "solution",
    type=str,
    help="The solution file to visualize",
    nargs="?",
    default="scheduling.sol",
)
parser.add_argument(
    "--no-z",
    help="Do not consider the z variables, just use start and completion times",
    action="store_true",
)
parser.add_argument(
    "--pdf",
    type=str,
    help="Write the plot to a PDF file",
    nargs="?",
    metavar="FILE",
)
args = parser.parse_args()

# Read the solution
values: dict[str, float] = {}
with open(args.solution, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        [name, value] = line.split(" ")
        values[name] = float(value)

# General comment: The completion time of a job is the last time step in which it is processed
# Similarily, the start time of a job is the first time step in which it is processed
# The duration is the number of time steps in which it is processed


def list2binstr(l: list[int]) -> str:
    return "".join(map(str, l))


# Create a dataframe with the job schedule
df = pd.DataFrame(
    columns=["job", "capacity", "machine", "start", "end", "duration", "zmask"]
)
for job in filter(lambda j: j != "0", milp.jobs):
    start = round(values[f"s_j_{job}"])
    end = round(values[f"c_j_{job}"])
    [assigned_machine] = [
        machine for machine in milp.machines if values[f"x_ik_{job}_{machine}"] >= 0.5
    ]
    capacity = milp.job_capacities[job]
    duration = end - start + 1
    all_zs = [
        [round(values[f"z_ikt_{job}_{machine}_{t}"]) for t in milp.timesteps]
        for machine in milp.machines
    ]
    [zs] = [z for z in all_zs if sum(z) > 0]
    zs = list2binstr(zs)
    df.loc[len(df)] = [job, capacity, assigned_machine, start, end, duration, zs]

print(df)

# Create patches for the legend
cmap = mpl.colormaps.get_cmap("tab10")
color_mapping = {
    m: cmap(i / (len(milp.machines) - 1)) for i, m in enumerate(milp.machines)
}
patches = []
for color in color_mapping.values():
    p = Patch(color=color)
    p.set_edgecolor("black")
    p.set_linewidth(1)
    patches.append(p)

# Create tick points (where a job starts or ends)
tick_points = df["start"].values.tolist() + (df["end"] + 1).values.tolist()
tick_points = list(set(tick_points))
tick_points.sort()

# Plot the jobs
# The grid lines are at the start of a time step. Hence, if a job ends in time step 11, the bar ends at 12.
fig, ax = plt.subplots()


def collect_binary_one_runs(s: str) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, c in enumerate(s):
        if c == "1":
            if start is None:
                start = i
            if i == len(s) - 1 or s[i + 1] == "0":
                runs.append((start, i - start + 1))
                start = None
    return runs


for i, row in df.iterrows():
    color = color_mapping[row["machine"]]
    bar_color = color if args.no_z else "none"
    padding = 0.1
    height = 1 - 2 * padding
    if not args.no_z:
        zruns = collect_binary_one_runs(row["zmask"])
        for zrun in zruns:
            ax.broken_barh(
                zruns, (i - 0.5 + padding, height), color=color_mapping[row["machine"]]
            )
    ax.barh(
        i,
        row["duration"],
        left=row["start"],
        height=height,
        edgecolor="black",
        linewidth=2,
        color=color,
    )

yticks = list(range(len(df)))
ax.set_yticks(yticks)
ax.set_yticklabels(df["job"])
ax.invert_yaxis()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
#plt.rc("font", family="serif")
plt.xlabel("Time")
plt.grid(axis="x", which="major")
plt.grid(axis="x", which="minor", alpha=0.4)
plt.legend(handles=patches, labels=color_mapping.keys())

if args.pdf:
    plt.tight_layout()
    plt.savefig(args.pdf, bbox_inches="tight")
else:
    plt.show()
