import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

class fano:
    def __init__(self, path_to_file: str):
        self.data = np.loadtxt(path_to_file)
        self.λmin = self.data[:,0].min()
        self.λmax = self.data[:,0].max()
        self.λ_fit = np.linspace(self.λmin, self.λmax, 1000)

    def lossless_model(self, λ, λ0, λ1, td, γ): # lossless transmission
            k = 2*np.pi / λ
            k0 = 2*np.pi / λ0
            k1 = 2*np.pi / λ1
            Γ = 2*np.pi / λ1**2 * γ
            t = td * (k - k0) / (k - k1 + 1j * Γ)
            return np.abs(t)**2
    
    def lossy_model(self, λ, λ0, λ1, td, γ, α): # lossy transmission
            k = 2*np.pi / λ
            k0 = 2*np.pi / λ0
            k1 = 2*np.pi / λ1
            Γ = 2*np.pi / λ1**2 * γ
            t = td * (k - k0 + 1j * α) / (k - k1 + 1j * Γ)
            return np.abs(t)**2

    def lossless_fit(self, code: str, fitting_params: list):
        def lossless_reflection(λ, λ0, λ1, td, γ):
            trans = self.lossless_model(λ, λ0, λ1, td, γ)
            return 1 - trans
        
        popt, pcov = curve_fit(self.lossless_model, self.data[:,0], self.data[:,1] , p0=fitting_params)

        plt.figure(figsize=(10,6))
        if code == "T":
            plt.plot(self.data[:,0], self.data[:,1], 'bo', label='Trans. data')
            plt.plot(self.λ_fit, self.lossless_model(self.λ_fit, *popt), 'cornflowerblue', label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f' % tuple(popt))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        if code == "R":
            plt.plot(self.data[:,0], 1-self.data[:,1], 'ro', label='Ref. data')
            plt.plot(self.λ_fit, lossless_reflection(self.λ_fit, *popt), 'darkred' , label = 'Reflectivity')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        if code == "both": 
            plt.plot(self.data[:,0], self.data[:,1], 'bo', label='Trans. data')
            plt.plot(self.data[:,0], 1-self.data[:,1], 'ro', label='Ref. data')
            plt.plot(self.λ_fit, lossless_reflection(self.λ_fit, *popt), color="darkred" , label = 'Reflectivity')
            plt.plot(self.λ_fit, self.lossless_model(self.λ_fit, *popt), color="cornflowerblue", label='Transmission (λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f)' % tuple(popt))
            plt.subplots_adjust(bottom=0.15)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
        else:
            pass
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflection and Transmission Coefficient')
        plt.show()

    def lossy_fit(self, code: str, fitting_params: list):
        popt, pcov = curve_fit(self.lossy_model, self.data[:,0], self.data[:,1], p0=fitting_params)

        plt.figure(figsize=(10,6))
        if code == "T": 
            plt.plot(self.data[:,0], self.data[:,1], 'bo', label='Transmission data')
            plt.plot(self.λ_fit, self.lossy_model(self.λ_fit, *popt), 'cornflowerblue', label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(popt))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        if code == "R":
            plt.plot(self.data[:,0], self.data[:,1], 'ro', label='Reflection data')
            plt.plot(self.λ_fit, self.lossy_model(self.λ_fit, *popt), 'darkred' , label = 'fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(popt))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        else:
            pass
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflection/Transmission Coeffiecient')
        plt.show()

    def theoretical_reflection_lossy(self, fitting_params: list):
        popt, pcov = curve_fit(self.lossy_model, self.data[:,0], self.data[:,1], p0=fitting_params)

        λ0s, λ1s, tds, γλs, αs = popt
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
        for λ_val in self.λ_fit:
            r_val = rds + (xbs + 1j * ybs) / (2 * np.pi / λ_val - 2 * np.pi / λ1s+ 1j * γs)
            r.append(r_val)
        t2 = tds + as_ / (2 * np.pi / λ1s - 2 * np.pi / λ1s + 1j * γs)
        r = np.array(r)
        reflectivity_values = np.abs(r)**2
        plt.figure(figsize=(10,6))
        plt.plot(self.λ_fit, reflectivity_values, 'b-', label='Reflectivity model')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflection Coeffiecient')
        plt.legend()
        plt.show()
        
########################################              Class documentation              ########################################
##
## The fano class takes a path to an appropriate transmission/reflectivity datafile as it's only input.
##
## Attributes: 
##
## The loaded data from the given input file is referred to simply as "self.data". 
## The minimum and maximum wavelength of the relevant interval is referred to as "self.λmin" and "self.λmax", respecitvely.
## The range between λmin and λmax is defined as "self.λ_fit", and is defines to be used in the methods described below. 
##
## Methods:
##
## fano.lossy_fit takes arguments code -> "R" / "T" refering to the type of data one wishes to fit (i.e. transmission or  
## reflectivity), and a list of initial guesses for the fitting parameters [λ0, λ1, td, γ, α].
##
## fano.lossless_fit takes the same arguments as fano.lossy_fit, except for the lack of α in the fitting parameters, and the  
## additional option of choosing to plot and fit both transmission and reflecitivity data (for this option set code -> "both").
##
## fano.lossless_fit only works for transmission data, while fano.lossy_fit can handle both transmission and reflectivity data 
## (plots/fit are produced according to the chosen code/type).

#ref = fano("/Users/mikkelodeon/optomechanics/Data/400um grating/reflectivity_400um_grating.txt")
trans = fano("/Users/mikkelodeon/optomechanics/400um gratings/01/Data/transmission_400um_grating.txt")
ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/01/Data/reflectivity_400um_grating.txt")

#grating.lossless_fit("both",[951, 951, 0.6, 2])

#ref.lossy_fit("R", [952, 952, 0.6, 1, 0.1])
#short_scan.lossy_fit("T", [951.9, 951.9, 0.30, 0.075, 0])
trans.theoretical_reflection_lossy([952, 952, 0.6, 1, 0.1])




    



