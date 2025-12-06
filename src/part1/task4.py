import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Load data
#call the dataset folder "datset5"
df = pd.read_csv("./dataset_part2/task4/robot_speed_task4.csv", header=None)
df.columns = ["distance_cm", "dt_s"]

print("\nLoaded data:")
print(df)

#  Each segment is 40 cm (fixed step)
dx = 40.0  # cm

# Compute segment speeds
df["speed_cm_s"] = dx / df["dt_s"]

# Mean speed
speed_mean = df["speed_cm_s"].mean()

print("\nSegment-wise speeds (cm/s):")
print(df[["distance_cm", "dt_s", "speed_cm_s"]])

print(f"\nEstimated AVERAGE robot speed: {speed_mean:.3f} cm/s")


#  Plot Δt values
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(df["distance_cm"], df["dt_s"], marker='o')
plt.title("Measured Δt for each 40 cm segment")
plt.xlabel("Distance marker (cm)")
plt.ylabel("Δt (s)")
plt.grid(True)

#  Plot speed for each segment
plt.subplot(1,2,2)
plt.scatter(df["distance_cm"], df["speed_cm_s"], marker='o', color='green')
plt.title("Estimated robot speed per segment")
plt.xlabel("Distance marker (cm)")
plt.ylabel("Speed (cm/s)")
plt.grid(True)

plt.tight_layout()
plt.show()

filename = "./src/part1/motor_calibration.csv"

with open(filename, "w") as f:
    f.write("average_speed_cm_per_s\n")
    f.write(f"{speed_mean}\n")

print("CSV file updated.")
