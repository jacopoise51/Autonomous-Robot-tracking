import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# Path to IMU log file (Task 1)
FILENAME = "./data/task1/imu_reading_task1.csv"

# Column names (the file NO header row)
col_names = [
    "timestamp",                     # time in milliseconds
    "acc_x", "acc_y", "acc_z",       # accelerometer [g]
    "roll", "pitch",                 # angles from accelerometer [deg]
    "gyro_x", "gyro_y", "gyro_z",    # gyroscope [deg/s]
    "mag_x", "mag_y", "mag_z"        # magnetometer [Gauss]
]

# Load CSV file
df = pd.read_csv(FILENAME, header=None, names=col_names)

# Build a relative time axis (start at t = 0)
t_raw = df["timestamp"].values
t = (t_raw - t_raw[0]) * 1e-3   # convert ms → s


# ---------------------------------------------------------
# TASK 1a — IMU Data Visualization
# ---------------------------------------------------------

# Color palettes
c_acc = ["#1f77b4", "#ff7f0e", "#2ca02c"]
c_gyro = ["#d62728", "#9467bd", "#8c564b"]
c_mag = ["#e377c2", "#7f7f7f", "#bcbd22"]

# ------------------ ACCELEROMETER -----------------------
plt.figure(figsize=(10, 6))
plt.suptitle("Task 1a – Accelerometer Readings (Static IMU)", fontsize=14)

plt.subplot(3, 1, 1)
plt.plot(t, df["acc_x"], color=c_acc[0])
plt.ylabel("acc_x [g]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["acc_y"], color=c_acc[1])
plt.ylabel("acc_y [g]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, df["acc_z"], color=c_acc[2])
plt.ylabel("acc_z [g]")
plt.xlabel("time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------ ROLL & PITCH -----------------------
plt.figure(figsize=(10, 4))
plt.suptitle("Task 1a – Roll and Pitch from Accelerometer", fontsize=14)

plt.subplot(2, 1, 1)
plt.plot(t, df["roll"], color="#17becf")
plt.ylabel("roll [deg]")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, df["pitch"], color="#bcbd22")
plt.ylabel("pitch [deg]")
plt.xlabel("time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------ GYROSCOPE --------------------------
plt.figure(figsize=(10, 6))
plt.suptitle("Task 1a – Gyroscope Readings (Static IMU)", fontsize=14)

plt.subplot(3, 1, 1)
plt.plot(t, df["gyro_x"], color=c_gyro[0])
plt.ylabel("gyro_x [deg/s]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["gyro_y"], color=c_gyro[1])
plt.ylabel("gyro_y [deg/s]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, df["gyro_z"], color=c_gyro[2])
plt.ylabel("gyro_z [deg/s]")
plt.xlabel("time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------ MAGNETOMETER -----------------------
plt.figure(figsize=(10, 6))
plt.suptitle("Task 1a – Magnetometer Readings", fontsize=14)

plt.subplot(3, 1, 1)
plt.plot(t, df["mag_x"], color=c_mag[0])
plt.ylabel("mag_x [Gauss]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["mag_y"], color=c_mag[1])
plt.ylabel("mag_y [Gauss]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, df["mag_z"], color=c_mag[2])
plt.ylabel("mag_z [Gauss]")
plt.xlabel("time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# TASK 1b — Gyroscope Bias and Variance
# ---------------------------------------------------------

gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]

# Bias = mean value during static IMU
gyro_bias = df[gyro_cols].mean()

# Variance = sample variance
gyro_var = df[gyro_cols].var(ddof=1)

print("=== Task 1b: Gyroscope Bias (mean) [deg/s] ===")
for axis in gyro_cols:
    print(f"{axis}: {gyro_bias[axis]:.6f}")

print("\n=== Task 1b: Gyroscope Variance [(deg/s)^2] ===")
for axis in gyro_cols:
    print(f"{axis}: {gyro_var[axis]:.6e}")

# Covariance matrix (optional)
gyro_cov = df[gyro_cols].cov()
print("\n=== Task 1b: Gyroscope Covariance Matrix ===")
print(gyro_cov)
