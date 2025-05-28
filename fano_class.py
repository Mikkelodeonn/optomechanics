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
    
    def lossy_fit(self, fitting_params: list, with_errors=False):

        popt, pcov = curve_fit(self.lossy_model, self.data[:,0], self.data[:,1], p0=fitting_params)

        if with_errors == True:
            errs = np.sqrt(np.diag(pcov))
            return [popt, errs]
        else:
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

rdata1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 ref.txt")
tdata1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")
rdata2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 ref.txt")
tdata2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
#tdata = fano("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/grating trans/M5_trans.txt")

rparams1 = rdata1.lossy_fit([951,951,0.1,0.5,1e-6])
tparams1 = tdata1.lossy_fit([951,951,0.1,0.5,1e-6])

rparams2 = rdata2.lossy_fit([951,951,0.1,0.5,1e-6])
tparams2 = tdata2.lossy_fit([951,951,0.1,0.5,1e-6])

print("rd1: ", rparams1[2])
print("rd2: ", rparams2[2])

#plt.figure(figsize=(10,7))

#plt.scatter(rdata1.data[:,0], rdata1.data[:,1], marker=".", color="lightcoral")#, label='$R_{G1}$')
#plt.plot(rdata1.λ_fit, rdata1.lossy_model(rdata1.λ_fit, *rparams1), 'lightcoral', alpha=0.4)#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.plot(rdata1.λ_fit, rdata1.lossy_model(rdata1.λ_fit, *rparams1), 'lightcoral', alpha=0.4)#, label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams1))
#plt.scatter(tdata1.data[:,0], tdata1.data[:,1], marker=".", color="skyblue")#, label='$T_{G1}$')
#plt.plot(tdata1.λ_fit, tdata1.lossy_model(tdata1.λ_fit, *tparams1), 'skyblue', alpha=0.4, label="$\\lambda_0 = $ %snm" % str(round(tparams1[0],2)))#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(tparams))
#plt.plot(tdata1.λ_fit, tdata1.lossy_model(tdata1.λ_fit, *tparams1), 'skyblue', alpha=0.4, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$m^{-1}$' % tuple(tparams1))

#plt.scatter(rdata2.data[:,0], rdata2.data[:,1], marker=".", color="darkred")#, label='$R_{G2}$')
#plt.plot(rdata2.λ_fit, rdata2.lossy_model(rdata2.λ_fit, *rparams2), 'darkred', alpha=0.4)#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.plot(rdata2.λ_fit, rdata2.lossy_model(rdata2.λ_fit, *rparams2), 'darkred', alpha=0.4)#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.scatter(tdata2.data[:,0], tdata2.data[:,1], marker=".", color="darkblue")#, label='$T_{G2}$')
#plt.plot(tdata2.λ_fit, tdata2.lossy_model(tdata2.λ_fit, *tparams2), 'darkblue', alpha=0.4, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$m^{-1}$' % tuple(tparams2))
#plt.plot(tdata2.λ_fit, tdata2.lossy_model(tdata2.λ_fit, *tparams2), 'darkblue', alpha=0.4, label="$\\lambda_0^{\\prime} = $ %snm" % str(round(tparams2[0],2)))#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(tparams))
#plt.xlabel("wavelength [nm]", fontsize=28)
#plt.ylabel("norm. ref./trans.", fontsize=28)
#plt.xticks(fontsize=21)
#plt.yticks(fontsize=21)
#plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
#plt.subplots_adjust(bottom=0.3)
#plt.grid(alpha=0.3)
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






    



