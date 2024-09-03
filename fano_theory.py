from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

## grating -> 400um, batch 06, number 01

grating = fano("/Users/mikkelodeon/optomechanics/400um gratings/01/Data/transmission_400um_grating.txt")

params = grating.lossy_fit([952,952,0.6,1,0.1])

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
    yb_initial_guess=0.5
    ybs = fsolve(equations,yb_initial_guess)

    r = []
    for λ_val in grating.λ_fit:
        r_val = rds + (xbs + 1j * ybs) / (2 * np.pi / λ_val - 2 * np.pi / λ1s+ 1j * γs)
        r.append(r_val)
    t2 = tds + as_ / (2 * np.pi / λ1s - 2 * np.pi / λ1s + 1j * γs)
    r = np.array(r)
    reflectivity_values = np.abs(r)**2

    return reflectivity_values

def theoretical_reflection_values_plot(reflection_values):
    plt.figure(figsize=(10,6))
    plt.plot(grating.λ_fit, reflectivity_values, 'b-', label='Calculated reflectivity')
    plt.plot(grating.λ_fit, grating.lossy_model(grating.λ_fit, *popt), 'r-', label='Transmission fit')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection Coeffiecient')
    plt.legend()
    plt.show()

def fano_cavity_transmission(params: list, code: str):
    reflection_values = theoretical_reflection_values(params)
    transmission_values = grating.lossy_model(grating.λ_fit, *params)

    if code == "cavitylength":
        def cavity_transmission(l):
            tg = np.min(transmission_values)
            rg = np.max(reflection_values)
            tm = 0.04
            rm = 0.96
            λ = params[0]
            T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 

        lengths = np.linspace(1,2000,10000)
        plt.figure(figsize=(10,6))
        plt.plot(lengths, fano_cavity_transmission(lengths))
        plt.show()

    if code == "wavelength":
        def cavity_transmission(λ, rg, tg):
            tm = 0.04
            rm = 0.96
            l = 1
            T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 

        λs = np.linspace(grating.data[:,0].min(), grating.data[:,0].max(), len(reflection_values))
        Ts = []
        for i in range(len(λs)):
            T = cavity_transmission(λs[i], reflection_values[i], transmission_values[i])
            Ts.append(T)

        plt.figure(figsize=(10,6))
        plt.plot(λs, Ts)
        plt.show()

def dual_fano_transmission(params: list, code: str):
    
    reflection_values = theoretical_reflection_values(params)
    transmission_values = grating.lossy_model(grating.λ_fit, *params)

    if code == "cavitylength":
        def cavity_transmission(l):
            λ = params[0]
            rg = np.max(reflection_values)
            tg = np.min(transmission_values)
            T = np.abs((tg**2)*np.exp(1j*(2*np.pi/λ)*l)/(1-(rg**2)*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 

        lengths = np.linspace(1,2000,10000)
        Ts = []
        #for i in range(len(λs)):
            #T = fano_cavity_transmission(λs[i], reflectivity_values[i], transmission_values[i])
            #Ts.append(T)

        plt.figure(figsize=(10,6))
        plt.plot(lengths, cavity_transmission(lengths))
        plt.show()

    if code == "wavelength":
        def cavity_transmission(λ, rg, tg):
            l = 1
            T = np.abs((tg**2)*np.exp(1j*(2*np.pi/λ)*l)/(1-(rg**2)*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 
        λs = np.linspace(grating.data[:,0].min(), grating.data[:,0].max(), len(reflection_values))
        Ts = []
        for i in range(len(λs)):
            T = cavity_transmission(λs[i], reflection_values[i], transmission_values[i])
            Ts.append(T)

        plt.figure(figsize=(10,6))
        plt.plot(λs, Ts)
        plt.show()


dual_fano_transmission(params,"wavelength")










