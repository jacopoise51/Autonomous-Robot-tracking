import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#  Load data
df = pd.read_csv("./dataset5/task4/robot_speed_task4.csv", header=None)
df.columns = ["distance_cm", "time_s"]

# Extract columns
x = df["distance_cm"].values.reshape(-1,1)   # distance (cm)
t = df["time_s"].values.reshape(-1,1)        # time (s)

#  Linear regression: x = v * t + b
model = LinearRegression().fit(t, x)
speed = model.coef_[0][0]      # slope = speed (cm/s)
bias  = model.intercept_[0]    # intercept (not important)


print(f"Estimated robot speed: {speed:.3f} cm/s")
print(f"Regression intercept: {bias:.3f} cm")


#  Plot distance vs time
plt.figure(figsize=(10,6))
plt.scatter(t, x, color='blue', label='Measured data')

# Regression line
t_fit = np.linspace(t.min(), t.max(), 100).reshape(-1,1)
x_fit = model.predict(t_fit)

plt.plot(t_fit, x_fit, color='red', label='Linear fit')

# Labels
plt.title("Robot Motion Calibration – Distance vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Distance (cm)")
plt.grid(True)
plt.legend()

plt.show()
