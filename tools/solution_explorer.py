import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# Import the problem description
import milp

# Read the solution
values: dict[str, float] = {}
with open("scheduling.sol") as f:
    for line in f:
        if line.startswith("#"):
            continue
        [name, value] = line.split(" ")
        values[name] = float(value)

# General comment: The completion time of a job is the last time step in which it is processed
# Similarily, the start time of a job is the first time step in which it is processed
# The duration is the number of time steps in which it is processed

# Create a dataframe with the job schedule
df = pd.DataFrame(columns=["job", "capacity", "machine", "start", "end", "duration"])
for job in filter(lambda j: j != "0", milp.jobs):
    start = round(values[f"s_j_{job}"])
    end = round(values[f"c_j_{job}"])
    [assigned_machine] = [
        machine for machine in milp.machines if values[f"x_ik_{job}_{machine}"] > 0.5
    ]
    capacity = milp.job_capacities[job]
    duration = end - start + 1
    df.loc[len(df)] = [job, capacity, assigned_machine, start, end, duration]

print(df)

# Create patches for the legend
cmap = mpl.colormaps.get_cmap("tab10")
color_mapping = {
    m: cmap(i / (len(milp.machines) - 1)) for i, m in enumerate(milp.machines)
}
patches = []
for color in color_mapping.values():
    patches.append(Patch(color=color))

# Create tick points (where a job starts or ends)
tick_points = df["start"].values.tolist() + (df["end"] + 1).values.tolist()
tick_points = list(set(tick_points))
tick_points.sort()

# Plot the jobs
# The grid lines are at the start of a time step. Hence, if a job ends in time step 11, the bar ends at 12.
plt.barh(
    df["job"],
    width=df["duration"],
    left=df["start"],
    color=df["machine"].map(color_mapping),
)
plt.gca().invert_yaxis()
plt.xlabel("Time")
plt.xticks(tick_points, minor=True)
plt.grid(axis="x", which="major")
plt.grid(axis="x", which="minor", alpha=0.4)
plt.legend(handles=patches, labels=color_mapping.keys())
plt.show()
