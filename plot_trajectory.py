import pandas as pd
import matplotlib.pyplot as plt
import os

folder = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    folder,
    "vehicle_trajectories.csv"
)

# Load trajectory data
df = pd.read_csv(csv_path)

print("Total trajectory points:", len(df))
print("Unique vehicles:", df["track_id"].nunique())

print("\nVehicles:")
print(
    df.groupby(["track_id", "vehicle"])
      .size()
      .sort_values(ascending=False)
)

# Plot each vehicle
plt.figure(figsize=(12, 7))

for track_id, vehicle_data in df.groupby("track_id"):

    plt.plot(
        vehicle_data["center_x"],
        vehicle_data["center_y"],
        marker=".",
        label=f"ID {track_id}"
    )

plt.gca().invert_yaxis()

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Vehicle Trajectories")

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()