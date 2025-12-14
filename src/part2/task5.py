import numpy as np
import pandas as pd
import math


# focal length f obtained from Task 3
f = 545.0224088789435

# True physical height of each QR code (in cm)
h0 = 11.5

# True robot position 
px_true = 60.2     # cm
py_true = 33.6     # cm
psi_true_deg = 90  # degrees
psi_true = np.deg2rad(psi_true_deg)  # convert to radians

# QR codes that were actually detected on Wall 2
valid_qr = [7, 8, 9, 1, 2, 3, 15, 14]

#Qr code global positions (in cm)

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


df = pd.read_csv("./dataset_part2/task5/camera_localization_task5.csv", header=None)
df.columns = ["timestamp", "qr_id", "Cx", "Cy", "width", "height", "raw_dist", "raw_angle"]

# Keep only the QR codes that are relevant for this experiment
df = df[df["qr_id"].isin(valid_qr)]

# Lists to store distance and angle errors
dist_errors = []
angle_errors = []

def wrap_angle(a):
    """Wrap angle 'a' into the range [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

#compute errors for each detection

for _, row in df.iterrows():

    qr = int(row["qr_id"])
    Cx = float(row["Cx"])
    h = float(row["height"])

    #5.1 Camera-derived measurements
    # Distance estimate from QR height in pixels
    d_cam = (h0 * f) / h

    # Angle estimate from QR center pixel location
    phi_cam = math.atan(Cx / f)

    #5.2 True QR code global position
    sx, sy = QR_GLOBAL[qr]

    #5.3 Ground-truth distance and angle
    d_true = math.sqrt((sx - px_true)**2 + (sy - py_true)**2)

    # global bearing to QR minus robot heading
    phi_true = math.atan2((sy - py_true), (sx - px_true)) - psi_true
    phi_true = wrap_angle(phi_true)

    #5.4 Measurement errors
    dist_errors.append(d_cam - d_true)
    angle_errors.append(wrap_angle(phi_cam - phi_true))

#Compute variances

sigma_d2 = np.var(dist_errors)
sigma_phi2 = np.var(angle_errors)

# Measurement noise covariance matrix (diagonal)
R = np.diag([sigma_d2, sigma_phi2])

print("Variance of distance measurements (σ_d^2):", sigma_d2)
print("Variance of angle measurements (σ_phi^2):", sigma_phi2)
print("\nMeasurement noise covariance matrix R:")
print(R)



measurements = []   # list of (d_cam, phi_cam)
qr_positions = []   # list of (sx, sy)

for _, row in df.iterrows():
    qr = int(row["qr_id"])
    Cx = float(row["Cx"])
    h = float(row["height"])

    d_cam = (h0 * f) / h
    phi_cam = math.atan(Cx / f)

    measurements.append([d_cam, phi_cam])
    qr_positions.append(QR_GLOBAL[qr])

measurements = np.array(measurements)
qr_positions = np.array(qr_positions)



def g(x):
    """Compute expected distances and angles given robot state x = [px, py, psi]."""
    px, py, psi = x
    preds = []

    for (sx, sy) in qr_positions:
        dx = sx - px
        dy = sy - py

        d_est = math.sqrt(dx*dx + dy*dy)
        phi_est = wrap_angle(math.atan2(dy, dx) - psi)

        preds.append([d_est, phi_est])

    return np.array(preds)



def jacobian(x):
    """  
    Compute the Jacobian matrix J(x) of g(x) with respect to x = [px, py, psi] for each QR detection.
    The output is a matrix of shape (2*M, 3), where M is the number of QR detections.
    """
    px, py, psi = x
    J = []

    for (sx, sy) in qr_positions:
        dx = sx - px
        dy = sy - py
        d = math.sqrt(dx*dx + dy*dy)

        # ∂d/∂px, ∂d/∂py
        dd_dpx = -dx / d
        dd_dpy = -dy / d
        dd_dpsi = 0

        # ∂phi/∂px, ∂phi/∂py, ∂phi/∂psi
        denom = dx*dx + dy*dy
        dphi_dpx = dy / denom
        dphi_dpy = -dx / denom
        dphi_dpsi = -1

        J.append([dd_dpx, dd_dpy, dd_dpsi])
        J.append([dphi_dpx, dphi_dpy, dphi_dpsi])

    return np.array(J)



def residuals_unweighted(x):
    """
    Compute the unweighted residual vector r(x).
    For each QR measurement we stack:
        [e_d, e_phi]
    where:
        e_d   = distance_error  = d_measured - d_predicted
        e_phi = angle_error     = phi_measured - phi_predicted (wrapped in [-pi, pi])
    The output is a vector of length 2*M (M = number of QR detections).
    """
    px, py, psi = x
    pred = g(x)
    res_list = []

    for i in range(len(pred)):
        e_d = measurements[i, 0] - pred[i, 0]
        e_phi = wrap_angle(measurements[i, 1] - pred[i, 1])
        res_list.append(e_d)
        res_list.append(e_phi)

    return np.array(res_list)


def cost_JWLS(x):
    """
    Compute the weighted least-squares cost:
        J(x) = Σ ( e_i^T R^{-1} e_i )
    where e_i is the (distance, angle) residual pair for each QR detection.
    Since R is the same for every measurement, R^{-1} is constant.
    This evaluates the true WLS cost, not the squared unweighted residuals.
    """
    res = residuals_unweighted(x)
    R_inv = np.linalg.inv(R)
    J = 0.0

    for i in range(0, len(res), 2):
        e_vec = np.array([res[i], res[i+1]])
        J += e_vec.T @ R_inv @ e_vec

    return J


#Levenberg–Marquardt Algorithm 

# Initial guess for the robot state [px, py, psi]
x = np.array([50.0, 40.0, np.deg2rad(85.0)])
print("\nInitial robot state guess:")
print("px =", x[0])
print("py =", x[1])
print("psi (deg) =", np.rad2deg(x[2]))

lambda_ = 1e-3     # initial damping factor
nu = 10            # factor for adapting lambda
max_iter = 50      # maximum number of LM iterations

R_inv = np.linalg.inv(R)

for it in range(max_iter):

    # Compute residuals and Jacobian at current estimate
    res = residuals_unweighted(x)   
    G = jacobian(x)                 

    # Build normal-equation components:
    #   H = G^T R^{-1} G
    #   gk = G^T R^{-1} e
    H = np.zeros((3, 3))
    gk = np.zeros(3)

    for i in range(0, len(res), 2):
        # Residual pair for this QR detection: [e_d, e_phi]
        e_vec = np.array([res[i], res[i+1]])

        # Corresponding 2 rows of the Jacobian (first distance, then angle)
        G_i = G[i:i+2, :]   # shape: 2 x 3

        # Accumulate weighted contributions
        H += G_i.T @ R_inv @ G_i
        gk += G_i.T @ R_inv @ e_vec

    # LM modification: (H + λ I) Δx = gk
    H_lm = H + lambda_ * np.eye(3)

    # Solve for the parameter update Δx (more stable than direct inversion)
    dx = np.linalg.solve(H_lm, gk)

    # New candidate estimate
    x_new = x + dx

    # Evaluate cost before and after the update
    J_old = cost_JWLS(x)
    J_new = cost_JWLS(x_new)

    # LM acceptance criterion: accept update only if cost decreases
    if J_new < J_old:
        x = x_new
        lambda_ /= nu   # decrease damping -> move closer to Gauss–Newton
        print(f"Iter {it}: accepted, J = {J_new:.4f}, lambda -> {lambda_}")
    else:
        lambda_ *= nu   # increase damping -> move closer to gradient descent
        print(f"Iter {it}: rejected, J = {J_new:.4f}, lambda -> {lambda_}")
    # Convergence condition: parameter update becomes very small
    if np.linalg.norm(dx) < 1e-6:
        print("Converged (small parameter update).")
        break

# Final estimated robot state
px_est, py_est, psi_est = x
print("\nEstimated robot state:")
print("px =", px_est)
print("py =", py_est)
print("psi (deg, unwrapped)  =", np.rad2deg(psi_est))
print("psi (deg, wrapped -180..180) =", (np.rad2deg(psi_est) + 180) % 360 - 180)
