import numpy as np
import pandas as pd
import math

# -------------------------------------------------------
# 1. PARAMETERS (KNOWN FROM CALIBRATION AND GROUND TRUTH)
# -------------------------------------------------------

# Insert your focal length f obtained from Task 3
f = 300.0   # <-- CHANGE THIS VALUE

# True physical height of each QR code (in cm)
h0 = 11.5

# True robot position (provided in readme.txt)
px_true = 60.2     # cm
py_true = 33.6     # cm
psi_true_deg = 90  # degrees
psi_true = np.deg2rad(psi_true_deg)  # convert to radians

# QR codes that were actually detected on Wall 2 (from readme.txt)
valid_qr = [7, 8, 9, 1, 2, 3, 15, 14]

# -------------------------------------------------------
# 2. GLOBAL POSITIONS OF QR CODES (FROM THE CSV FILE)
# -------------------------------------------------------
# You already extracted these values from qr_code_position_in_global_coordinate.csv

QR_GLOBAL = {
    1:  (50.5, 121.5),
    2:  (62.5, 121.5),
    3:  (74.5, 121.5),
    7:  (111.0, 121.5),
    8:  (98.0, 121.5),
    9:  (86.5, 121.5),
    14: (25.0, 121.5),
    15: (37.0, 121.5),
}

# -------------------------------------------------------
# 3. LOAD camera_localization_task5.csv
# -------------------------------------------------------

df = pd.read_csv("./dataset_part2/task5/camera_localization_task5.csv", header=None)
df.columns = ["timestamp", "qr_id", "Cx", "Cy", "width", "height", "raw_dist", "raw_angle"]

# Keep only the QR codes that are relevant for this experiment
df = df[df["qr_id"].isin(valid_qr)]

# Lists to store distance and angle errors
dist_errors = []
angle_errors = []

# -------------------------------------------------------
# 4. SUPPORT FUNCTION
# -------------------------------------------------------

def wrap_angle(a):
    """Wrap angle 'a' into the range [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

# -------------------------------------------------------
# 5. COMPUTE ERRORS FOR EACH QR MEASUREMENT
# -------------------------------------------------------

for _, row in df.iterrows():

    qr = int(row["qr_id"])
    Cx = float(row["Cx"])
    h = float(row["height"])

    # ----- 5.1 Camera-derived measurements -----
    # Distance estimate from QR height in pixels
    d_cam = (h0 * f) / h

    # Angle estimate from QR center pixel location
    phi_cam = math.atan(Cx / f)

    # ----- 5.2 True QR code global position -----
    sx, sy = QR_GLOBAL[qr]

    # ----- 5.3 Ground-truth distance and angle -----
    d_true = math.sqrt((sx - px_true)**2 + (sy - py_true)**2)

    # global bearing to QR minus robot heading
    phi_true = math.atan2((sy - py_true), (sx - px_true)) - psi_true
    phi_true = wrap_angle(phi_true)

    # ----- 5.4 Measurement errors -----
    dist_errors.append(d_cam - d_true)
    angle_errors.append(wrap_angle(phi_cam - phi_true))

# -------------------------------------------------------
# 6. COMPUTE VARIANCE MATRIX R
# -------------------------------------------------------

sigma_d2 = np.var(dist_errors)
sigma_phi2 = np.var(angle_errors)

# Measurement noise covariance matrix (diagonal)
R = np.diag([sigma_d2, sigma_phi2])

print("Variance of distance measurements (σ_d^2):", sigma_d2)
print("Variance of angle measurements (σ_phi^2):", sigma_phi2)
print("\nMeasurement noise covariance matrix R:")
print(R)
