import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Path to IMU log file
#call the dataset folder "datset5"
FILENAME = "./dataset5/task1/imu_reading_task1.csv"  # change path if needed


# 1. Load data and name columns


# The file has NO header row, so we set header=None and assign column names manually.
col_names = [
    "timestamp",     # [ms] according to description 
    "acc_x", "acc_y", "acc_z",      # linear acceleration [g]
    "roll", "pitch",                # [deg] from accelerometer
    "gyro_x", "gyro_y", "gyro_z",   # gyroscope [deg/s]
    "mag_x", "mag_y", "mag_z"       # magnetometer [Gauss]
]

df = pd.read_csv(FILENAME, header=None, names=col_names)

# Build a relative time axis starting from t = 0
# If timestamp is in milliseconds, divide by 1000 to get seconds.
t_raw = df["timestamp"].values
t = (t_raw - t_raw[0]) * 1e-3   # [s] 


# 2. Task 1a: Visualize the data


#Accelerometer (x, y, z) 
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, df["acc_x"])
plt.ylabel("acc_x [g]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["acc_y"])
plt.ylabel("acc_y [g]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, df["acc_z"])
plt.ylabel("acc_z [g]")
plt.xlabel("time [s]")
plt.grid(True)

plt.suptitle("Task 1a – Accelerometer readings (static IMU)")
plt.tight_layout()
plt.show()

# Roll & Pitch 
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, df["roll"])
plt.ylabel("roll [deg]")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, df["pitch"])
plt.ylabel("pitch [deg]")
plt.xlabel("time [s]")
plt.grid(True)

plt.suptitle("Task 1a – Roll and pitch from accelerometer")
plt.tight_layout()
plt.show()

#Gyroscope (x, y, z) 
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, df["gyro_x"])
plt.ylabel("gyro_x [deg/s]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["gyro_y"])
plt.ylabel("gyro_y [deg/s]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, df["gyro_z"])
plt.ylabel("gyro_z [deg/s]")
plt.xlabel("time [s]")
plt.grid(True)

plt.suptitle("Task 1a – Gyroscope readings (static IMU)")
plt.tight_layout()
plt.show()

# Magnetometer (x, y, z)
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, df["mag_x"])
plt.ylabel("mag_x [Gauss]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["mag_y"])
plt.ylabel("mag_y [Gauss]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, df["mag_z"])
plt.ylabel("mag_z [Gauss]")
plt.xlabel("time [s]")
plt.grid(True)

plt.suptitle("Task 1a – Magnetometer readings")
plt.tight_layout()
plt.show()


# 3. Task 1b: Bias and variance of gyroscope

gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]

# Bias = mean value over static interval
gyro_bias = df[gyro_cols].mean()

# Variance = sample variance over static interval
gyro_var = df[gyro_cols].var(ddof=1)  # ddof=1 → unbiased estimator

print("-- Task 1b: Gyroscope bias (mean) [deg/s] --")
for axis in gyro_cols:
    print(f"{axis}: {gyro_bias[axis]:.6f}")

print("\n-- Task 1b: Gyroscope variance [ (deg/s)^2 ] --")
for axis in gyro_cols:
    print(f"{axis}: {gyro_var[axis]:.6e}")

# covariance matrix
gyro_cov = df[gyro_cols].cov()
print("\n --Task 1b: Gyroscope covariance matrix --")
print(gyro_cov)
