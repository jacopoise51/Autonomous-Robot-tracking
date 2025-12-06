import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



#velocity gain 
k_v = 5.44 / 0.30     

# Gyroscope bias 
gyro_bias_z = 0.15198819

# Initial robot pose
px0 = 41.7     # cm
py0 = 18.5     # cm
phi0 = 0.0     # rad

x0 = np.array([px0, py0, phi0])

# True physical height of QR code 
h0 = 11.5

# Focal length f (from Task 3 calibration)
f = 545.0224088789435

# Global positions of QR codes 
QR_GLOBAL = {
    19: (0, 11),
    20: (0, 22.5),
    21: (0, 35),
    25: (0, 47),
    26: (0, 59.5),
    27: (0, 71),
    31: (0, 84),
    32: (0, 96),
    33: (0, 108),

    13: (13, 121.5),
    14: (25, 121.5),
    15: (37, 121.5),
    1:  (50.5, 121.5),
    2:  (62.5, 121.5),
    3:  (74.5, 121.5),
    9:  (86.5, 121.5),
    8:  (98, 121.5),
    7:  (111, 121.5),

    34: (13.5, 0),
    35: (26, 0),
    36: (38, 0),
    28: (50, 0),
    29: (61.5, 0),
    30: (73.5, 0),
    4:  (85.5, 0),
    5:  (97.6, 0),
    6:  (109.8, 0),

    10: (121.5, 12),
    11: (121.5, 24.2),
    12: (121.5, 36),
    16: (121.5, 49),
    17: (121.5, 61),
    18: (121.5, 72),
    22: (121.5, 85.5),
    23: (121.5, 98),
    24: (121.5, 109.5),
}


# Measurement noise covariance R (from Task 5 variance estimation) 
sigma_d2   = 18.05          # variance of distance [cm^2]
sigma_phi2 = 1.99e-4        # variance of angle [rad^2]
R = np.diag([sigma_d2, sigma_phi2])

# Process noise      
q_w  = 0.034361388405623215  



def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def motion_model(x, v, omega, dt):
    """
    Discrete-time unicycle-like motion model.
    State: x = [px, py, phi]
    v     : linear speed (cm/s)
    omega : angular rate (rad/s)
    """
    px, py, phi = x
    px_new  = px + dt * v * math.cos(phi)
    py_new  = py + dt * v * math.sin(phi)
    phi_new = wrap_angle(phi + dt * omega)
    return np.array([px_new, py_new, phi_new])

def motion_jacobian_F(x, v, omega, dt):
    """Jacobian F = d f / d x of the motion model."""
    px, py, phi = x
    F = np.eye(3)
    F[0, 2] = -dt * v * math.sin(phi)  # ∂px/∂phi
    F[1, 2] =  dt * v * math.cos(phi)  # ∂py/∂phi
    return F

def measurement_model_single_qr(x, sx, sy):
    """
    Camera measurement model for a single QR code with global position (sx, sy).
    Output: [d_pred, phi_rel_pred]
      d_pred       : distance from robot to QR
      phi_rel_pred : bearing of QR in camera frame (global bearing - robot heading)
    """
    px, py, phi = x
    dx = sx - px
    dy = sy - py
    d_pred = math.sqrt(dx*dx + dy*dy)
    bearing_global = math.atan2(dy, dx)
    phi_rel_pred = wrap_angle(bearing_global - phi)
    return np.array([d_pred, phi_rel_pred])

def measurement_jacobian_H(x, sx, sy):
    """
    Jacobian H = d h / d x for a single QR measurement.
    h(x) = [d, phi_rel]^T
    """
    px, py, phi = x
    dx = sx - px
    dy = sy - py
    r2 = dx*dx + dy*dy
    r  = math.sqrt(r2)

    # Avoid division by zero
    if r < 1e-6:
        r = 1e-6
        r2 = r*r

    # ∂d/∂px, ∂d/∂py, ∂d/∂phi
    dd_dpx = -dx / r
    dd_dpy = -dy / r
    dd_dphi = 0.0

    # ∂phi_rel/∂px, ∂phi_rel/∂py, ∂phi_rel/∂phi
    dphi_dpx  =  dy / r2
    dphi_dpy  = -dx / r2
    dphi_dphi = -1.0

    H = np.array([
        [dd_dpx,  dd_dpy,  dd_dphi],
        [dphi_dpx, dphi_dpy, dphi_dphi]
    ])
    return H

def process_noise_Q(dt, gyro_variance_z):
    """
    Euler–Maruyama process noise for nonlinear stochastic dynamics.
    Q_k = dt * B Σ_w B^T  
    
    gyro_variance_z : variance of gyro noise 
    """
    return dt * np.diag([0, 0, gyro_variance_z])

def dead_reckoning(t_imu, v, gyro_z, x0):
    """
    Dead-reckoning integration of IMU + PWM.
    Returns array of robot states over time.
    """
    N = len(t_imu)
    x_dr = np.zeros((N, 3))
    x_dr[0] = x0

    for k in range(1, N):
        dt = t_imu[k] - t_imu[k-1]
        x_dr[k] = motion_model(x_dr[k-1], v[k], gyro_z[k], dt)

    return x_dr

def ekf_predict(x, P, v, omega, dt, gyro_variance_z):
    """
    EKF prediction step using Euler–Maruyama process noise.
    """
    x_pred = motion_model(x, v, omega, dt)
    F = motion_jacobian_F(x, v, omega, dt)
    Q = process_noise_Q(dt, gyro_variance_z)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x, P, qr_id, Cx, h, QR_GLOBAL, f, h0, R):
    """
    EKF update for a single camera measurement.
    Uses QR global position and camera geometry.
    """
    if qr_id not in QR_GLOBAL:
        return x, P

    sx, sy = QR_GLOBAL[qr_id]

    # Camera-derived measurement
    d_meas = (h0 * f) / h
    phi_meas = math.atan(Cx / f)
    z = np.array([d_meas, phi_meas])

    # Predicted measurement
    z_pred = measurement_model_single_qr(x, sx, sy)

    # Innovation
    y = z - z_pred
    y[1] = wrap_angle(y[1])

    # Jacobian
    H = measurement_jacobian_H(x, sx, sy)

    # EKF update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x_upd = x + K @ y
    x_upd[2] = wrap_angle(x_upd[2])

    P_upd = (np.eye(3) - K @ H) @ P

    return x_upd, P_upd

def run_ekf(t_imu, v, gyro_z, df_cam, QR_GLOBAL, x0, P0, f, h0, R, gyro_variance_z):
    """
    Run EKF over entire IMU + camera dataset.
    Returns filtered trajectory and covariances.
    """
    N = len(t_imu)
    x_ekf = np.zeros((N, 3))
    P_ekf = np.zeros((N, 3, 3))

    x_ekf[0] = x0
    P_ekf[0] = P0

    cam_idx = 0
    N_cam = len(df_cam)

    for k in range(1, N):
        dt = t_imu[k] - t_imu[k-1]

        # Prediction
        x_pred, P_pred = ekf_predict(
            x_ekf[k-1], P_ekf[k-1], v[k], gyro_z[k], dt, gyro_variance_z
        )

        # Process all camera measurements up to this IMU time
        while cam_idx < N_cam and df_cam.iloc[cam_idx]["t"] <= t_imu[k]:
            row = df_cam.iloc[cam_idx]
            cam_idx += 1

            x_pred, P_pred = ekf_update(
                x_pred, P_pred,
                int(row["qr_id"]), float(row["Cx"]), float(row["height"]),
                QR_GLOBAL, f, h0, R
            )

        x_ekf[k] = x_pred
        P_ekf[k] = P_pred

    return x_ekf, P_ekf


# IMU
df_imu = pd.read_csv(
    "./dataset_part2/task6-task7/imu_tracking_task6.csv",
    header=None,
    names=[
        "timestamp", "acc_x", "acc_y", "acc_z",
        "roll", "pitch",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z"
    ]
)

# PWM
df_pwm = pd.read_csv(
    "./dataset_part2/task6-task7/motor_control_tracking_task6.csv",
    header=None,
    names=["timestamp", "pwm_left", "pwm_right"]
)

# CAMERA
df_cam = pd.read_csv(
    "./dataset_part2/task6-task7/camera_tracking_task6.csv",
    header=None,
    names=["timestamp", "qr_id", "Cx", "Cy", "width", "height", "raw_dist", "raw_angle"]
)

# Sort all by timestamp (just to be safe)
df_imu = df_imu.sort_values("timestamp").reset_index(drop=True)
df_pwm = df_pwm.sort_values("timestamp").reset_index(drop=True)
df_cam = df_cam.sort_values("timestamp").reset_index(drop=True)

imu_ts = df_imu["timestamp"].values
pwm_ts = df_pwm["timestamp"].values
cam_ts = df_cam["timestamp"].values

# Shared time origin
t0 = min(imu_ts[0], pwm_ts[0], cam_ts[0])
t_imu = imu_ts - t0
t_pwm = pwm_ts - t0
t_cam = cam_ts - t0

# Correct gyro_z 
gyro_z = (df_imu["gyro_z"].values - gyro_bias_z) * np.pi / 180.0

# Reindex PWM onto IMU timeline with zero-order hold, fill NaN with 0
df_pwm.index = t_pwm
df_pwm = df_pwm.drop(columns=["timestamp"])
df_pwm_resampled = df_pwm.reindex(t_imu, method="ffill").fillna(0.0)

pwm_left  = df_pwm_resampled["pwm_left"].values
pwm_right = df_pwm_resampled["pwm_right"].values

pwm_avg = 0.5 * (pwm_left + pwm_right)
v = k_v * pwm_avg   # [cm/s]

# Camera times
df_cam["t"] = t_cam

x_dr = dead_reckoning(t_imu, v, gyro_z, x0)

x_ekf, P_ekf = run_ekf(
    t_imu, v, gyro_z, df_cam, QR_GLOBAL,
    x0,
    np.diag([1.0, 1.0, np.deg2rad(5)**2]),
    f, h0, R, q_w
)



# Plotting results
plt.figure(figsize=(8, 8))

# walls 
plt.axvline(0,      color='k', linestyle='--', linewidth=1)
plt.axvline(121.5,  color='k', linestyle='--', linewidth=1)
plt.axhline(0,      color='k', linestyle='--', linewidth=1)
plt.axhline(121.5,  color='k', linestyle='--', linewidth=1)

# QR-code positions
for qr_id, (sx, sy) in QR_GLOBAL.items():
    plt.plot(sx, sy, 'ko')
    plt.text(sx+1, sy+1, str(qr_id), fontsize=8)

# Dead-reckoning trajectory
plt.plot(x_dr[:,0], x_dr[:,1], 'b--', label="Dead reckoning (IMU + PWM)")

# EKF trajectory
plt.plot(x_ekf[:,0], x_ekf[:,1], 'r', label="EKF (IMU + Camera)")

plt.xlabel("px [cm]")
plt.ylabel("py [cm]")
plt.title("Task 7 – Tracking with IMU and Camera (EKF)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()
