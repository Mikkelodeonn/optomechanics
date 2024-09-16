from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

## grating -> 400um, batch 06, M3

M1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M1/400_M1 trans.txt")
M2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M2/400_M2 trans.txt")
M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M4 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M4/400_M4 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

params1 = M3.lossy_fit([952,952,0.6,1,0.1])
params2 = M3.lossy_fit([952,952,0.6,1,0.1])
params2[1] = params2[1] 
params2[0] = params2[0] + 0.3

#params1 = [950, 950.3, 0.81, 0.48, 9e-7]
#params2 = [949.8, 950, 0.81, 0.48, 9e-7]
## grating parameters -> [λ0, λ1, td, γλ, α]
# λ0 -> resonance wavelength
# λ1 -> guided mode resonance wavelength
# td -> direct transmission coefficient
# γλ -> width of guided mode resonance
# α  -> loss factor
#Δ = 0.7 #nm
#grating1 = [950, 950, 0.81, 0.48, 9e-7]
#grating2 = [950+Δ, 950+Δ, 0.81, 0.48, 9e-7]   

λs_range = np.linspace(951,953,1000)

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

def resonant_cavity_length(params: list):
    λs = np.linspace(900, 1100, 10000) 
    reflection_values = theoretical_reflection_values(params)[1]
    transmission_values = np.sqrt(model(λs, *params))

    lengths = []
    Ts = []

    ls = np.linspace(0,1000,10000)

    for l in ls:
        tg = np.min(transmission_values)
        rg = np.max(reflection_values)
        tm = np.sqrt(0.04)
        rm = np.sqrt(0.96)
        λ = params[0]
        t = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
        Ts.append(t)

    peak_indices = find_peaks(Ts)

    for idx in peak_indices[0]:
        lengths.append(ls[idx])
    return lengths[0]
 
def fano_cavity_transmission(params: list):
    reflection_values = theoretical_reflection_values(params)[1]
    transmission_values = np.sqrt(model(λs_range, *params))

    length = resonant_cavity_length(params)

    def cavity_transmission(λ, rg, tg, l):
        tm = np.sqrt(0.04)
        rm = np.sqrt(0.96)
        T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
        return T 

    λs = np.linspace(λs_range.min(), λs_range.max(), len(reflection_values))
    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values[i], transmission_values[i], length)
        Ts.append(T)

    return (λs,Ts)

def fano_cavity_transmission_plot(params: list):
    λs, Ts = fano_cavity_transmission(params)
    plt.figure(figsize=(10,6))
    plt.plot(λs, Ts)
    plt.title("Single fano cavity transmission as function of wavelength (l = %sμm)" % str(round(length,2)))
    plt.xlabel("Wavelength [nm]") 
    plt.ylabel("Intensity [arb.u.]")
    plt.show()

def dual_fano_transmission(params1: list, params2: list, l: float):
    
    reflection_values1 = theoretical_reflection_values(params1)[1]
    transmission_values1 = np.sqrt(model(λs_range, *params1))
    reflection_values2 = theoretical_reflection_values(params2)[1]
    transmission_values2 = np.sqrt(model(λs_range, *params2))

    def cavity_transmission(λ, rg1, tg1, rg2, tg2, l):
        T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
        return T 
    λs = np.linspace(λs_range.min(), λs_range.max(), len(reflection_values1))
    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values1[i], transmission_values1[i], reflection_values2[i], transmission_values2[i], l)
        Ts.append(T)

    return (λs, Ts)

def dual_fano_transmission_plot(params1: list, params2: list, length: float):
    λs, Ts =  dual_fano_transmission(params1, params2, length)
    plt.figure(figsize=(10,6))
    plt.title("Dual fano cavity transmission as a function of wavelength (l = %sμm)" % str(round(length,2)))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
    plt.plot(λs, Ts)
    plt.show()

def detuning_plot(Δs: list, length: float): ## plots dual fano cavity transmission for different values for the detuning
    plt.figure(figsize=(10,6))
    for Δ in Δs:
        grating1 = [950, 950, 0.81, 0.48, 9e-7]
        grating2 = [950+Δ, 950+Δ, 0.81, 0.48, 9e-7]
        λs, Ts =  dual_fano_transmission(grating1, grating2, length)
        plt.plot(λs, Ts, label="Δ=%snm" %(Δ))

    plt.title("Dual fano cavity transmission as a function of wavelength (l = %sμm)" % str(round(length,2)))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
    plt.xlim(949.75,950.75)
    plt.legend()
    plt.show()

def cavity_length_plot(ls: list, params1: list, params2: list):
    plt.figure(figsize=(15,6))
    for l in ls:
        λs, Ts = dual_fano_transmission(params1, params2, l)
        plt.plot(λs, Ts, label="cavity length: %sμm" % str(round(l,2)))
    plt.title("Dual fano cavity transmission as a function of wavelength")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.70)
    plt.show() 

Δs = [0.01, 0.04, 0.08, 0.12, 0.20, 0.30, 0.40] # detuning in nm    
ls = np.linspace(resonant_cavity_length(params1), resonant_cavity_length(params2),20)
length = resonant_cavity_length(params2)

#detuning_plot(Δs, length)
cavity_length_plot(ls, params1, params2)
#λs1, Ts1 =  fano_cavity_transmission(params2)
#λs2, Ts2 =  dual_fano_transmission(params1, params2, length)
#plt.figure(figsize=(10,6))
#plt.plot(λs1, Ts1, label="Reg. Fano")
#plt.plot(λs2, Ts2, label="Double Fano")
#plt.legend()
#plt.show()
#dual_fano_transmission_plot(params1, params2, length)



