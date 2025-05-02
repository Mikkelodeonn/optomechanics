import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Step 1: Define the Gaussian intensity profile
def gaussian_intensity(x, beam_waist):
    return np.exp(-2 * (x / beam_waist) ** 2)

# Step 2: Parameters for the beam and razor blade
beam_waist = 1.0  # Beam waist (radius at 1/e^2 intensity)
x = np.linspace(-3 * beam_waist, 3 * beam_waist, 1000)  # x-axis for visualization

# Step 3: Simulate the razor blade and compute transmitted power
blade_positions = np.linspace(-3 * beam_waist, 3 * beam_waist, 100)  # Blade positions
transmitted_power = []

for blade_pos in blade_positions:
    # Integrate the Gaussian intensity from the blade position to infinity
    power, _ = quad(lambda x: gaussian_intensity(x, beam_waist), blade_pos, np.inf)
    transmitted_power.append(power)

# Step 4: Plot the Gaussian beam profile and transmitted power
plt.figure(figsize=(12, 6))

# Plot 1: Gaussian beam profile with a razor blade
plt.subplot(1, 2, 1)
plt.plot(x, gaussian_intensity(x, beam_waist), label="gaussian intensity")
plt.axvline(0, color="red", linestyle="--", label="beam Center")
#plt.title("Gaussian Beam Profile")
plt.xlabel("position (x)")
plt.ylabel("intensity")
plt.legend()
#plt.grid()

# Plot 2: Transmitted power vs blade position
plt.subplot(1, 2, 2)
plt.plot(blade_positions, transmitted_power, label="transmitted power")
#plt.title("Transmitted Power vs Razor Blade Position")
plt.xlabel("blade position (x)")
plt.ylabel("transmitted power")
plt.legend()
#plt.grid()

plt.tight_layout()
plt.show()