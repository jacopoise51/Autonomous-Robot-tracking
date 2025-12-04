import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Task 3a: Camera module calibration

# 1. Settings

# CSV file for Task 3
#call the dataset folder "datset5"
csv_file = "./dataset_part2/task3/camera_module_calibration_task3.csv"   

# From readme.txt:
dist_camera_pinhole_to_IR = 1.7   # [cm]
dist_wall_to_wooden_list   = 5.0  # [cm]

# Total offset to get TRUE distance between QR-code plane and camera pinhole
total_offset_cm = dist_camera_pinhole_to_IR + dist_wall_to_wooden_list


# 2. Load calibration data

col_names = ["distance_measured_cm", "height_px"]
df = pd.read_csv(csv_file, header=None, names=col_names)

# Extract as numpy arrays
d_measured = df["distance_measured_cm"].to_numpy()   # [cm]
h_pixels   = df["height_px"].to_numpy()              # [pixel]

# Compute TRUE distance between camera pinhole and QR code
d_true = d_measured + total_offset_cm                # [cm]

# Inverse height (1/h) for linear relation
inv_h = 1.0 / h_pixels                               # [1/pixel]


# 3. Plot 1/height vs TRUE distance
plt.figure(figsize=(6, 4))
plt.scatter(d_true, inv_h, label="data")
plt.xlabel("True distance camera–to–QR [cm]")
plt.ylabel("1 / height [1/pixel]")
plt.title("Task 3a – 1/height vs true distance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 4. Linear regression: distance = k * (1/h) + b

# fit the model: d_true ≈ k * (1/h) + b

k, b = np.polyfit(inv_h, d_true, 1)   # returns slope k and intercept b

print("-- Task 3a: Linear regression results --")
print("Model: distance ≈ k * (1/height) + b")
print(f"Gradient k = {k:.6f}  [cm·pixel]")
print(f"Bias     b = {b:.6f}  [cm]")


# 5. Plot distance vs 1/height with fitted line

inv_h_line = np.linspace(inv_h.min(), inv_h.max(), 200)
d_fit = k * inv_h_line + b

plt.figure(figsize=(6, 4))
plt.scatter(inv_h, d_true, label="data")
plt.plot(inv_h_line, d_fit, 'r', linewidth=2, label="linear fit")
plt.xlabel("1 / height [1/pixel]")
plt.ylabel("True distance camera–to–QR [cm]")
plt.title("Task 3a – Linear fit: distance vs 1/height")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# 4. Task 3b: Compute focal length f in pixels

h0_cm = 11.5   # [cm] true height of the QR-code
f_pixels = k / h0_cm

print("\n-- Task 3b: Focal length estimation ")
print(f"QR-code true height h0 = {h0_cm:.2f} cm")
print(f"Focal length f ≈ {f_pixels:.3f} pixels")



#WRITE RESULTS TO FILE IF CHANGED

filename = "./src/part1/camera_results.csv"

with open(filename, "w") as f:
    f.write("slope,bias,focal_pixels,focal_cm\n")
    f.write(f"{k},{b},{f_pixels},{k}\n")

print("CSV file updated.")

