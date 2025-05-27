import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Step 1: Define the Gaussian intensity profile
def gaussian_intensity(x, beam_waist):
    return np.exp(-x**2 / (2*beam_waist**2))

# Step 2: Parameters for the beam and razor blade
FWHM = 300
beam_waist = FWHM/2.35482  # Beam waist ->
x = np.linspace(-350, 350, 10000)  # x-axis for visualization

# Step 3: Simulate the razor blade and compute transmitted power
blade_positions = np.linspace(-2 * beam_waist, 1 * beam_waist, 100)  # Blade positions
transmitted_power = []

#for blade_pos in blade_positions:
    # Integrate the Gaussian intensity from the blade position to infinity
power1, _ = quad(lambda x: gaussian_intensity(x, beam_waist), -200, -100)
power2, _ = quad(lambda x: gaussian_intensity(x, beam_waist), -50, 50)
total_power, _ = quad(lambda x: gaussian_intensity(x, beam_waist), -np.inf, np.inf)
    #transmitted_power.append(power)

# Step 4: Plot the Gaussian beam profile and transmitted power
plt.figure(figsize=(10, 7))

# Plot 1: Gaussian beam profile with a razor blade
#plt.subplot(1, 2, 1)
plt.plot(x, gaussian_intensity(x, beam_waist), label="gaussian")
plt.plot([-200]*10, np.linspace(0,1,10), linestyle="--", color="firebrick")
plt.plot([-100]*10, np.linspace(0,1,10), linestyle="--", color="firebrick")
plt.plot([-50]*10, np.linspace(0,1,10), linestyle="--", color="forestgreen")
plt.plot([50]*10, np.linspace(0,1,10), linestyle="--", color="forestgreen")
plt.fill_between(np.linspace(-200,-100,100), [0]*100, [1]*100, color="firebrick", alpha=0.3, label="trans.: %s%%" % str(round(power1/total_power*1e2,2)))
plt.fill_between(np.linspace(-50,50,100), [0]*100, [1]*100, color="forestgreen", alpha=0.3, label="trans.: %s%%" % str(round(power2/total_power*1e2,2)))
#plt.axvline(0, color="firebrick", linestyle="--", label="beam center")
#plt.title("Gaussian Beam Profile")
plt.xlabel("position [μm]", fontsize=28)
plt.ylabel("norm. trans.", fontsize=28)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)

#plt.grid()

# Plot 2: Transmitted power vs blade position
#plt.subplot(1, 2, 2)
#plt.plot(blade_positions, transmitted_power, label="transmitted power")
#plt.title("Transmitted Power vs Razor Blade Position")
#plt.xlabel("blade position (x)", fontsize=28)
#plt.ylabel("trans. power", fontsize=28)
#plt.legend()
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
#plt.grid()

#plt.tight_layout()
plt.show()