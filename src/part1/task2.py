import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#  Load IMU calibration data (Task 2)


df = pd.read_csv(r"./data/task2/imu_calibration_task2.csv")


# Extract accelerometer axes (in g units)
ax = df.iloc[:,1]
ay = df.iloc[:,2]
az = df.iloc[:,3]


#  Function to extract UP and DOWN values for a given axis

def extract_up_down(signal, threshold=0.8):
    """
    Detects the UP (+1g) and DOWN (-1g) sections of the accelerometer signal.
    
    Parameters:
        signal    : 1-D array of acceleration values for one axis (ax, ay, or az)
        threshold : threshold to detect ±1g plateaus (default: 0.5g)

    Returns:
        au : mean acceleration during UP (+1g) period
        ad : mean acceleration during DOWN (-1g) period
    """

    # UP region: values significantly above 0 (approx. +1g)
    up_mask = signal > threshold

    # DOWN region: values significantly below 0 (approx. -1g)
    down_mask = signal < -threshold

    # Compute means for UP and DOWN regions
    au = signal[up_mask].mean()
    ad = signal[down_mask].mean()

    return au, ad


#  Extract UP and DOWN means for each axis


ax_u, ax_d = extract_up_down(ax)
ay_u, ay_d = extract_up_down(ay)
az_u, az_d = extract_up_down(az)

#  Compute gain and bias for each axis


g = 1  

def compute_gain_bias(au, ad):
    """
    Computes gain (k) and bias (b) using the calibration formulas:

        k = (a_u - a_d) / (2g)
        b = (a_u + a_d) / 2
    """
    k = (au - ad) / (2 * g)
    b = (au + ad) / 2
    return k, b

# Calculate gain and bias for all 3 axes
kx, bx = compute_gain_bias(ax_u, ax_d)
ky, by = compute_gain_bias(ay_u, ay_d)
kz, bz = compute_gain_bias(az_u, az_d)


#  Print results


print("========================================================")
print(" RAW UP/DOWN MEANS (g units) ")
print("========================================================")
print(f"AX: a_u = {ax_u:.5f}, a_d = {ax_d:.5f}")
print(f"AY: a_u = {ay_u:.5f}, a_d = {ay_d:.5f}")
print(f"AZ: a_u = {az_u:.5f}, a_d = {az_d:.5f}")

print("\n========================================================")
print(" GAIN AND BIAS RESULTS ")
print("========================================================")
print(f"X-axis: gain = {kx:.6f}, bias = {bx:.6f}")
print(f"Y-axis: gain = {ky:.6f}, bias = {by:.6f}")
print(f"Z-axis: gain = {kz:.6f}, bias = {bz:.6f}")


# Plot accelerometer data


plt.figure(figsize=(12,6))
plt.plot(ax, label='ax')
plt.plot(ay, label='ay')
plt.plot(az, label='az')
plt.axhline(0.8, color='black', linestyle='--', alpha=0.4)
plt.axhline(-0.8, color='black', linestyle='--', alpha=0.4)
plt.title("Accelerometer readings (Task 2 calibration)")
plt.xlabel("Samples")
plt.ylabel("Acceleration (g)")
plt.legend()
plt.show()
