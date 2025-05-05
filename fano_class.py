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

        return popt
    
    def lossy_fit(self, fitting_params: list):

        popt, pcov = curve_fit(self.lossy_model, self.data[:,0], self.data[:,1], p0=fitting_params)

        return popt
    
    def lossless_fit_plot(self, code: str, params: list):

        popt = self.lossy_fit(params)

        plt.figure(figsize=(10,6))
        if code == "T":
            plt.plot(self.data[:,0], self.data[:,1], 'bo', label='Trans. data')
            plt.plot(self.λ_fit, self.lossless_model(self.λ_fit, *popt), 'cornflowerblue', label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f' % tuple(popt))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        if code == "R":
            plt.plot(self.data[:,0], 1-self.data[:,1], 'ro', label='Ref. data')
            plt.plot(self.λ_fit, (self.lossless_model(self.λ_fit, *popt)-1), 'darkred' , label = 'Reflectivity')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        if code == "both": 
            plt.plot(self.data[:,0], self.data[:,1], 'bo', label='Trans. data')
            plt.plot(self.data[:,0], 1-self.data[:,1], 'ro', label='Ref. data')
            plt.plot(self.λ_fit, (self.lossless_model(self.λ_fit, *popt)-1), color="darkred" , label = 'Reflectivity')
            plt.plot(self.λ_fit, self.lossless_model(self.λ_fit, *popt), color="cornflowerblue", label='Transmission (λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f)' % tuple(popt))
            plt.subplots_adjust(bottom=0.15)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
        else:
            pass
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflection and Transmission Coefficient')
        plt.show()

    def lossy_fit_plot(self, code: str, params: str):
        
        popt = self.lossy_fit(params)

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

rdata = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 ref.txt")
tdata = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")

rparams = rdata.lossy_fit([951,951,0.1,0.5,1e-6])
tparams = tdata.lossy_fit([951,951,0.1,0.5,1e-6])

plt.figure(figsize=(10,6))
plt.scatter(rdata.data[:,0], rdata.data[:,1], marker=".", color="firebrick", label='ref. data')
plt.plot(rdata.λ_fit, rdata.lossy_model(rdata.λ_fit, *rparams), 'firebrick', alpha=0.5, label="ref. fit")#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
plt.scatter(tdata.data[:,0], tdata.data[:,1], marker=".", color="blueviolet", label='trans. data')
plt.plot(tdata.λ_fit, tdata.lossy_model(tdata.λ_fit, *tparams), 'cornflowerblue', alpha=0.5, label="trans. fit")#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(tparams))
plt.legend(loc='upper right')
plt.xlabel("wavelength [nm]")
plt.ylabel("normalized ref./trans. [arb. u.]")
#plt.show()
#fitting_params = [951.2,951.2,0.1,0.04,1e-6]
#data = fano("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/Data/1293um/1293 short.txt")
#data = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M7/400_M7 ref.txt")
#params = data.lossy_fit(fitting_params)
#rparams = R.lossy_fit(fitting_params)

#plt.figure(figsize=(10,7))

#plt.title("1293μm double fano cavity (M3+M5)") 
#plt.plot(data.data[:,0], data.data[:,1], 'bo', label='data')
#plt.plot(data.λ_fit, data.lossy_model(data.λ_fit, *params), label="fit: linewidth=%spm \nexpected linewidth: ~40pm" % str(round(2*np.abs(params[3]),4)*1e3))
#plt.plot(data.λ_fit, data.lossy_model(data.λ_fit, *params), 'cornflowerblue', label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(params))

#plt.plot(R.data[:,0], R.data[:,1], 'ro', label='Reflection data')
#plt.plot(R.λ_fit, R.lossy_model(R.λ_fit, *rparams), 'darkred', label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
#plt.show()






    



