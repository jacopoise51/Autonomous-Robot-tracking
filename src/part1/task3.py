import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_file = "./data/task3/camera_module_calibration_task3.csv"   # adjust path if needed

dist_camera_pinhole_to_IR = 1.7   # [cm]
dist_wall_to_wooden_list = 5.0    # [cm]

# Total offset that must be added to the measured distances
total_offset_cm = dist_camera_pinhole_to_IR + dist_wall_to_wooden_list



# 2. Load calibration data


col_names = ["distance_measured_cm", "height_px"]
df = pd.read_csv(csv_file, header=None, names=col_names)

# Convert to numpy arrays
d_measured = df["distance_measured_cm"].to_numpy()   # [cm]
h_pixels   = df["height_px"].to_numpy()              # [px]

# Compute TRUE distance from camera pinhole to QR plane
d_true = d_measured + total_offset_cm                # [cm]


inv_h = 1.0 / h_pixels



# 3. Plot: 1/height vs TRUE distance

plt.figure(figsize=(6, 4))
plt.scatter(d_true, inv_h, color="#1f77b4", s=60, label="Measured data")
plt.xlabel("True distance camera → QR [cm]")
plt.ylabel("1 / QR height [1/pixel]")
plt.title("Task 3a – Inverse height vs true distance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# 4. Linear regression:  d_true = k * (1/h) + b


# Fit model: distance ≈ k*(1/h) + b
k, b = np.polyfit(inv_h, d_true, 1)

print("=== Task 3a: Linear regression results ===")
print("Model: distance ≈ k * (1/height) + b")
print(f"Slope    k = {k:.6f}   [cm · pixel]")
print(f"Offset   b = {b:.6f}   [cm]")



# 5. Plot regression line 


inv_h_line = np.linspace(inv_h.min(), inv_h.max(), 200)
d_fit = k * inv_h_line + b

plt.figure(figsize=(6, 4))
plt.scatter(inv_h, d_true, color="#ff7f0e", s=60, label="Measured data")
plt.plot(inv_h_line, d_fit, color="black", linewidth=2, label="Linear fit")
plt.xlabel("1 / QR height [1/pixel]")
plt.ylabel("True distance camera → QR [cm]")
plt.title("Task 3a – Linear fit: distance vs 1/height")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# 6. Task 3b – Compute focal length f (in pixels)
h0_cm = 11.5
f_pixels = k / h0_cm

print("\n=== Task 3b: Focal length estimation ===")
print(f"QR-code height h0 = {h0_cm:.2f} cm")
print(f"Estimated focal length f ≈ {f_pixels:.3f} pixels")

