import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# velocity gain (from Task 4: avg speed = 5.44 cm/s at PWM = 0.30)
k_v = 5.44 / 0.30     

# Gyroscope bias 
gyro_bias_z = 0.15198819

# Initial robot pose 
px0 = 41.7     # cm  
py0 = 18.5     # cm
phi0 = 0.0     # rad


# IMU and PWM data loading

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

# Extract raw timestamps
imu_ts = df_imu["timestamp"].values
pwm_ts = df_pwm["timestamp"].values

# Shared time origin
t0 = min(imu_ts[0], pwm_ts[0])

t_imu = imu_ts - t0
t_pwm = pwm_ts - t0

# Gyro bias correction
gyro_z = (df_imu["gyro_z"].values - gyro_bias_z) * np.pi/180.0

# Reindex PWM with global time
df_pwm.index = t_pwm
df_pwm = df_pwm.drop(columns=["timestamp"])

# Zero-order hold to IMU times
df_pwm_resampled = df_pwm.reindex(t_imu, method="ffill")
df_pwm_resampled = df_pwm_resampled.fillna(0.0)


pwm_left  = df_pwm_resampled["pwm_left"].values
pwm_right = df_pwm_resampled["pwm_right"].values


# Velocity from PWM
pwm_avg = 0.5 * (pwm_left + pwm_right)
v = k_v * pwm_avg     # cm/s



# linearized  state update function

def f(x, v, omega, dt):
    """
    Discrete-time quasi-constant turn model (Euler-Maruyama).
    Inputs:
        x     = [px, py, phi]
        v     = measured forward velocity (odometry)
        omega = measured angular velocity (gyro)
        dt    = sampling time
    """
    px, py, phi = x
    px_new  = px + dt * v * np.cos(phi)
    py_new  = py + dt * v * np.sin(phi)
    phi_new = phi + dt * omega
    return np.array([px_new, py_new, phi_new])



#dead reckoning

N = len(t_imu)
x_est = np.zeros((N, 3))
x_est[0] = np.array([px0, py0, phi0])

for k in range(1, N):
    dt = t_imu[k] - t_imu[k-1]
    x_est[k] = f(x_est[k-1], v[k], gyro_z[k], dt)
    # Wrap angle to [-pi, pi]
    x_est[k, 2] = (x_est[k, 2] + np.pi) % (2*np.pi) - np.pi




plt.figure(figsize=(7,7))
plt.plot(x_est[:,0], x_est[:,1], label="Dead reckoning (IMU + PWM ZOH)")
plt.xlabel("px [cm]")
plt.ylabel("py [cm]")
plt.title("Task 6 – Dead-Reckoning Trajectory")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.axvline(0, color='k', linestyle='--')       # Wall 1
plt.axvline(121.5, color='k', linestyle='--')   # Wall 3
plt.axhline(0, color='k', linestyle='--')       # Wall 4
plt.axhline(121.5, color='k', linestyle='--')   # Wall 2
plt.show()



