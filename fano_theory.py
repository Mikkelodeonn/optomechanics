from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

## grating -> 400um, batch 06, M3

M1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M1/400_M1 trans.txt")
M2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M2/400_M2 trans.txt")
M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M4 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M4/400_M4 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

params1 = M3.lossy_fit([952,952,0.6,1,0.1])
params2 = M5.lossy_fit([952,952,0.6,1,0.1])
#print(params)

## grating parameters -> [λ0, λ1, td, γλ, α]
# λ0 -> resonance wavelength
# λ1 -> guided mode resonance wavelength
# td -> direct transmission coefficient
# γλ -> width of guided mode resonance
# α  -> loss factor
#Δ = 0.7 #nm
#grating1 = [950, 950, 0.81, 0.48, 9e-7]
#grating2 = [950+Δ, 950+Δ, 0.81, 0.48, 9e-7]   

λs_range = np.linspace(949,954,1000)

def model(λ, λ0, λ1, td, γ, α): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    Γ = 2*np.pi / λ1**2 * γ
    t = td * (k - k0 + 1j * α) / (k - k1 + 1j * Γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list):
    λ0s, λ1s, tds, γλs, αs = params
    γs = 2*np.pi / λ1s**2 * γλs
    as_ = tds * (2*np.pi / λ1s - 2*np.pi / λ0s - 1j*γs)
    xas = np.real(as_)
    yas = np.imag(as_)

    rds = np.sqrt(1 - tds**2)
    xbs = -xas * tds / rds
    As = 0.015 * (γs**2 + (2*np.pi/λ0s - 2*np.pi/λ1s)**2)

    def equations(vars):
        yb = vars
        return xas**2 + yas**2 + xbs**2 + yb**2 + 2 * γs * rds * yb + 2 * γs * tds * yas + As
    yb_initial_guess = 0.5
    ybs = fsolve(equations,yb_initial_guess)

    r = []
    for λ_val in λs_range:
        r_val = rds + (xbs + 1j * ybs) / (2 * np.pi / λ_val - 2 * np.pi / λ1s+ 1j * γs)
        r.append(r_val)
    r = np.array(r)
    reflectivity_values = np.abs(r)**2
    complex_reflectivity_values = r

    return (reflectivity_values, complex_reflectivity_values)

def theoretical_reflection_values_plot(reflection_values):
    plt.figure(figsize=(10,6))
    plt.plot(λs_range, reflection_values, 'b-', label='Calculated reflectivity')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection Coeffiecient')
    plt.legend()
    plt.show()
 
def fano_cavity_transmission(params: list, code: str):
    reflection_values = theoretical_reflection_values(params)[1]
    transmission_values = np.sqrt(model(λs_range, *params))

    if code == "cavitylength":
        def cavity_transmission(l):
            tg = np.min(transmission_values)
            rg = np.max(reflection_values)
            tm = np.sqrt(0.04)
            rm = np.sqrt(0.96)
            λ = params[0]
            T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 

        lengths = np.linspace(1,2000,10000)
        plt.figure(figsize=(10,6))
        plt.plot(lengths, cavity_transmission(lengths))
        plt.title("Single fano cavity transmission as function of cavity length")
        plt.xlabel("Cavity length [μm]") 
        plt.ylabel("Intensity [arb.u.]")
        plt.show()

    if code == "wavelength":
        def cavity_transmission(λ, rg, tg):
            tm = np.sqrt(0.04)
            rm = np.sqrt(0.96)
            l = 1
            T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 

        λs = np.linspace(λs_range.min(), λs_range.max(), len(reflection_values))
        Ts = []
        for i in range(len(λs)):
            T = cavity_transmission(λs[i], reflection_values[i], transmission_values[i])
            Ts.append(T)

        plt.figure(figsize=(10,6))
        plt.plot(λs, Ts)
        plt.title("Single fano cavity transmission as function of wavelength")
        plt.xlabel("Wavelength [nm]") 
        plt.ylabel("Intensity [arb.u.]")
        plt.show()

def dual_fano_transmission(params1: list, params2: list, code: str):
    
    reflection_values1 = theoretical_reflection_values(params1)[1]
    transmission_values1 = np.sqrt(model(λs_range, *params1))
    reflection_values2 = theoretical_reflection_values(params2)[1]
    transmission_values2 = np.sqrt(model(λs_range, *params2))

    if code == "cavitylength":
        def cavity_transmission(l):
            λ = (params1[0] + params2[0])/2
            rg1 = np.max(reflection_values1)
            tg1 = np.min(transmission_values1)
            rg2 = np.max(reflection_values2)
            tg2 = np.min(transmission_values2)
            T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 

        lengths = np.linspace(1,2000,10000)

        return (lengths, cavity_transmission(lengths))

    if code == "wavelength":
        def cavity_transmission(λ, rg1, tg1, rg2, tg2):
            l = 20
            T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 
        λs = np.linspace(λs_range.min(), λs_range.max(), len(reflection_values1))
        Ts = []
        for i in range(len(λs)):
            T = cavity_transmission(λs[i], reflection_values1[i], transmission_values1[i], reflection_values2[i], transmission_values2[i])
            Ts.append(T)

        return (λs, Ts)

def detuning_plot(Δs): ## plots dual fano cavity transmission for different values for the detuning
    plt.figure(figsize=(10,6))
    for Δ in Δs:
        grating1 = [950, 950, 0.81, 0.48, 9e-7]
        grating2 = [950+Δ, 950+Δ, 0.81, 0.48, 9e-7]
        λs, Ts =  dual_fano_transmission(grating1, grating2, code="wavelength")
        plt.plot(λs, Ts, label="Δ=%snm" %(Δ))

    plt.title("Dual fano cavity transmission as a function of wavelength (cavity length: 10μm)")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
    plt.legend()
    plt.show()

def dual_fano_transmission_plot(params1: list, params2: list, code: str):
    λs, Ts =  dual_fano_transmission(params1, params2, code=code)
    plt.figure(figsize=(10,6))
    plt.title("Dual fano cavity transmission as a function of wavelength")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
    plt.plot(λs, Ts)
    plt.show()

Δs = [0.01, 0.04, 0.07, 0.10, 0.20, 0.30, 0.40] # detuning in nm    

#detuning_plot(Δs)
dual_fano_transmission_plot(params1, params2, code="wavelength")
