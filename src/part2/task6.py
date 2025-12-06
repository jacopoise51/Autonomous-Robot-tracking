import numpy as np
import pandas as pd
import math


def f(x, v, omega, dt):
    """
    Discrete-time quasi-constant turn model (Euler–Maruyama).
    Inputs:
        x     = [px, py, phi]
        v     = measured forward velocity (odometry)
        omega = measured angular velocity (gyro)
        dt    = sampling time
    """
    px, py, phi = x

    px_next  = px + dt * v * np.cos(phi)
    py_next  = py + dt * v * np.sin(phi)
    phi_next = phi + dt * omega  # noise is added separately through Q

    return np.array([px_next, py_next, phi_next])


def process_noise_cov(dt, sigma_w):
    """
    Process noise covariance Q_k = dt * B_w Σ_w B_w^T
    Noise acts only on phi dynamics.
    """
    return dt * sigma_w**2 * np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

def pwm_to_velocity(pwm_left, pwm_right):
    # PWM are normalized between 0 and 1
    pwm_avg = 0.5 * (pwm_left + pwm_right)

    # k_v computed from calibration: 5.44 cm/s at PWM = 0.30
    k_v = 5.44 / 0.30   # = 18.1333 cm/s

    return k_v * pwm_avg   # cm/s
