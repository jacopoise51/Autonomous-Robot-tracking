import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Task 3a: Camera module calibration

# 1. Settings

# CSV file for Task 3
csv_file = "./dataset5/task3/camera_module_calibration_task3.csv"   # adjust path if needed

# From readme.txt:
# Distances that must be ADDED to the recorded distances
dist_camera_pinhole_to_IR = 1.7   # [cm]
dist_wall_to_wooden_list   = 5.0  # [cm]

# Total offset to get TRUE distance between QR-code plane and camera pinhole
total_offset_cm = dist_camera_pinhole_to_IR + dist_wall_to_wooden_list


# 2. Load calibration data
# File has two columns:
#  1) measured distance (cm)
#  2) detected QR-code height (px)

col_names = ["distance_measured_cm", "height_px"]
df = pd.read_csv(csv_file, header=None, names=col_names)

# Extract as numpy arrays
d_measured = df["distance_measured_cm"].to_numpy()   # [cm]
h_pixels   = df["height_px"].to_numpy()              # [pixel]

# Compute TRUE distance between camera pinhole and QR code
d_true = d_measured + total_offset_cm                # [cm]

# Inverse height (1/h) for linear relation
inv_h = 1.0 / h_pixels                               # [1/pixel]


# 3. Plot 1/height vs TRUE distance (as in the guide / Fig. 4)
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

# We fit the model:
#     d_true ≈ k * (1/h) + b
# This matches Eq. (3): x3 = h0 * f * (1/h) + b
# where k = h0 * f  (Task 3b will use this)
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

# ------------------------------------------------------------
# 6. (For your report)
# ------------------------------------------------------------
# You will report:
#   - The plot of 1/height vs distance
#   - The linear fit plot
#   - The numeric values of:
#       * k (gradient)
#       * b (bias)
#
# ------------------------------------------------------------
# 4. Task 3b: Compute focal length f in pixels
# ------------------------------------------------------------
# From Eq. (3) and the fitted model:
#   d_true = (h0 * f) * (1/h) + b
# so:
#   k = h0 * f  =>  f = k / h0
h0_cm = 11.5   # [cm] true height of the QR-code
f_pixels = k / h0_cm

print("\n-- Task 3b: Focal length estimation ")
print(f"QR-code true height h0 = {h0_cm:.2f} cm")
print(f"Focal length f ≈ {f_pixels:.3f} pixels")

# For your report, you can write something like:
# "Using linear regression on distance vs 1/height, we obtained
#  gradient k = ... cm·pixel and bias b = ... cm. With QR-code height
#  h0 = 11.5 cm, the focal length is estimated as f = k / h0 ≈ ... pixels."