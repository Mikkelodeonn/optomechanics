from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list, λs: np.array, losses=True, loss_factor=0.05):
    λ0s, λ1s, tds, γλs, βs = params
    γs = 2*np.pi / λ1s**2 * γλs
    a = tds * ((2*np.pi / λ1s) - (2*np.pi / λ0s) + 1j*βs - 1j*γs)
    xas = np.real(a)
    yas = np.imag(a)

    if losses == True:
        L = loss_factor
    if losses == False:
        L = 0

    c_squared = L * (γs**2 + (2*np.pi/λ0s - 2*np.pi/λ1s)**2)

    rds = np.sqrt(1 - tds**2)
    xbs = -(xas * tds / rds)

    def equations(vars):
        yb = vars
        return xas**2 + yas**2 + xbs**2 + yb**2 + 2 * γs * rds * yb + 2 * γs * tds * yas + c_squared
    yb_initial_guess = 0.5
    ybs = fsolve(equations, yb_initial_guess)

    r = []
    for λ_val in λs:
        r_val = rds + (xbs + 1j * ybs) / (2 * np.pi / λ_val - 2 * np.pi / λ1s+ 1j * γs)
        r.append(r_val)
    r = np.array(r)
    reflectivity_values = np.abs(r)**2
    complex_reflectivity_amplitudes = r

    return (reflectivity_values, complex_reflectivity_amplitudes)

M3 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M3_trans.txt")
M3_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M3_trans_PI.txt")
M3_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans.txt")
M3_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans_PI.txt")

M5 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M5_trans.txt")
M5_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M5_trans_PI.txt")
M5_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans.txt")
M5_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans_PI.txt")

M3[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M3[:,1], M3_PI[:,1], M3_norm[:,1], M3_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
M5[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5[:,1], M5_PI[:,1], M5_norm[:,1], M5_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

λs = np.linspace(M3[:,0][0], M3[:,0][-1], 50)
λs_fit = np.linspace(M3[:,0][0], M3[:,0][-1], 10000)

p0 = [952,952,0.6,1,0.1]
params1, pcov1 = curve_fit(model, M3[:,0], M3[:,1], p0=p0)
params2, pcov2 = curve_fit(model, M5[:,0], M5[:,1], p0=p0)

rs_M3 = theoretical_reflection_values(params1, λs)[0]
rs_M5 = theoretical_reflection_values(params2, λs)[0]
rs_M3 = [float(r) for r in rs_M3]
rs_M5 = [float(r) for r in rs_M5]

ts_M3 = model(λs, *params1)
ts_M5 = model(λs, *params2)

popt_t1, _ = curve_fit(model, λs, ts_M3, p0=[952,952,0.6,1,1e-7])
popt_t2, _ = curve_fit(model, λs, ts_M5, p0=[952,952,0.6,1,1e-7])

popt_r1, _ = curve_fit(model, λs, rs_M3, p0=[952,952,0.6,1,1e-7])
popt_r2, _ = curve_fit(model, λs, rs_M5, p0=[952,952,0.6,1,1e-7])

λt = np.array([0.5*params1[0] + 0.5*params2[0]])

t_M3_trans = model(λt, *params1)
t_M5_trans = model(λt, *params2)

r_M3_trans = theoretical_reflection_values(params1, λt)[0][0]
r_M5_trans = theoretical_reflection_values(params2, λt)[0][0]

print(t_M3_trans, t_M5_trans, r_M3_trans, r_M5_trans)

print(popt_t2)

plt.figure(figsize=(10,6))
plt.title("M3+M5 spectra in optimized double fano config. $\\left(\\lambda_{trans} = \\frac{(\\lambda_{M3} + \\lambda_{M5})}{2}\\right)$")
plt.plot(λs, ts_M3, "o", color="darkred", label="M3 trans.")
plt.plot(λs, rs_M3, "o", color="darkblue", label="M3 ref.")
plt.plot(λs, ts_M5, "o", color="firebrick", label="M5 trans.", alpha=0.6)
plt.plot(λs, rs_M5, "o", color="royalblue", label="M5 ref.", alpha=0.6)
plt.plot(λs_fit, model(λs_fit, *popt_r1), color="darkblue", label="$r_{M3,trans} = $" + str(round(r_M3_trans[0]*1e2,2)) + "%")
plt.plot(λs_fit, model(λs_fit, *popt_r2), color="royalblue", alpha=0.6, label="$r_{M5,trans} = $" + str(round(r_M5_trans[0]*1e2,2)) + "%")
plt.plot(λs_fit, model(λs_fit, *popt_t1), color="darkred", label="$t_{M3,trans} = $" + str(round(t_M3_trans[0]*1e2,3)) + "%")
plt.plot(λs_fit, model(λs_fit, *popt_t2), color="firebrick", alpha=0.6, label="$t_{M5,trans} = $" + str(round(t_M5_trans[0]*1e2,2)) + "%")
plt.ylabel("normalized transmission/reflection [arb. u.]")
plt.xlabel("wavelength [nm]")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.subplots_adjust(right=0.70)
#plt.show()