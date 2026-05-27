import numpy as np
import matplotlib.pyplot as plt

# 1. Physical Parameters
v0 = 20.0       # m/s
angle_deg = 45.0
g = 9.81        # m/s^2

# 2. Convert to Radians (crucial for np.sin/cos)
theta = np.radians(angle_deg)

# 3. Create time array (0 to 3 seconds, 100 points)
t = np.linspace(0, 3, 100)

# 4. Calculate X and Y (The Physics)
# FILL IN THE EQUATIONS HERE:
x = v0 * np.cos(theta) * t
y = v0 * np.sin(theta) * t - g * t**2 / 2

# 5. Plotting (This is how we "Architects" visualize results)
plt.plot(x, y)
plt.title("Projectile Motion")
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.grid(True)
plt.show()