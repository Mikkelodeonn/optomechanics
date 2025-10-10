import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib
from matplotlib.ticker import FuncFormatter
matplotlib.rcParams["font.sans-serif"] = "Times New Roman"
matplotlib.rcParams["font.family"] = "sans-serif"

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

M1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M1/400_M1 trans.txt")
M2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M2/400_M2 trans.txt")
M4 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M4/400_M4 trans.txt")
M7 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M7/400_M7 trans.txt")
M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

M1ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M1/400_M1 ref.txt")
M2ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M2/400_M2 ref.txt")
M4ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M4/400_M4 ref.txt")
M7ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M7/400_M7 ref.txt")
M3ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 ref.txt")
M5ref = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 ref.txt")


tparamsM1 = M1.lossy_fit([952,952,0.6,1,0.1])
tparamsM2 = M2.lossy_fit([952,952,0.6,1,0.1])
tparamsM4 = M4.lossy_fit([952,952,0.6,1,0.1])
tparamsM7 = M7.lossy_fit([952,952,0.6,1,0.1])
tparamsM3, tparamsM3_errs = M3.lossy_fit([952,952,0.6,1,0.1], with_errors=True)
tparamsM5, tparamsM5_errs = M5.lossy_fit([952,952,0.6,1,0.1], with_errors=True)

rparamsM1 = M1ref.lossy_fit([952,952,0.6,1,1e-6])
rparamsM2 = M2ref.lossy_fit([952,952,0.6,1,1e-6])
rparamsM4 = M4ref.lossy_fit([952,952,0.6,1,1e-6])
rparamsM7 = M7ref.lossy_fit([952,952,0.6,1,1e-6])
rparamsM3, rparamsM3_errs = M3ref.lossy_fit([952,952,0.6,1,1e-6], with_errors=True)
rparamsM5, rparamsM5_errs = M5ref.lossy_fit([952,952,0.6,1,1e-6], with_errors=True)

#rdata1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 ref.txt")
#tdata1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")
#rdata2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 ref.txt")
#tdata2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
#tdata = fano("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/grating trans/M5_trans.txt")

#rparams1 = rdata1.lossy_fit([951,951,0.1,0.5,1e-6])
#tparams1 = tdata1.lossy_fit([951,951,0.1,0.5,1e-6])

##rparams2 = rdata2.lossy_fit([951,951,0.1,0.5,1e-6])
#tparams2 = tdata2.lossy_fit([951,951,0.1,0.5,1e-6])

#print("rd1: ", rparams1[2])
#print("rd2: ", rparams2[2])

rparams = rparamsM3
tparams = tparamsM3
λ_fit = M3ref.λ_fit

rdata = M3ref.data
rfit = M3ref.lossy_model(λ_fit, *rparams)
rfit_p = M3.lossy_model(λ_fit, *rparams+rparamsM3_errs)
rfit_m  = M3.lossy_model(λ_fit, *rparams-rparamsM3_errs)

tdata = M3.data
tfit = M3.lossy_model(λ_fit, *tparams)
tfit_p = M3.lossy_model(λ_fit, *tparams+tparamsM3_errs)
tfit_m  = M3.lossy_model(λ_fit, *tparams-tparamsM3_errs)

rparams2 = rparamsM5
tparams2 = tparamsM5
λ_fit2 = M5ref.λ_fit

rdata2 = M5ref.data
rfit2 = M5ref.lossy_model(λ_fit2, *rparams2)
rfit2_p = M3.lossy_model(λ_fit, *rparams+rparamsM3_errs)
rfit2_m  = M3.lossy_model(λ_fit, *rparams-rparamsM3_errs)

tdata2 = M5.data
tfit2 = M5.lossy_model(λ_fit2, *tparams2)
tfit2_p = M3.lossy_model(λ_fit, *tparams+tparamsM5_errs)
tfi2_m  = M3.lossy_model(λ_fit, *tparams-tparamsM5_errs)

print("params of M3:", tparams, "+/-", tparamsM3_errs)
print("params of M5:", tparams2, "+/-", tparamsM5_errs)

fig, ax = plt.subplots(figsize=(11,7))

#ax2 = ax.twinx()

#plt.scatter(rdata2[:,0], rdata2[:,1], marker=".", color="indianred")#, label='$R_{G1}$')
#plt.plot(λ_fit2, rfit2, 'indianred', alpha=0.5)#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.plot(λ_fit, rdata1.lossy_model(rdata1.λ_fit, *rparams1), 'lightcoral', alpha=0.4)#, label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams1))
#plt.scatter(tdata2[:,0], tdata2[:,1], marker=".", color="skyblue")#, label='$T_{G1}$')
#plt.plot(λ_fit2, tfit2, 'skyblue', alpha=0.5, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$nm^{-1}$' % tuple(tparams2))

ax.scatter(rdata[:,0], rdata[:,1], marker=".", color="deepskyblue")#, label='$R_{G2}$')
ax.plot(λ_fit, rfit, 'deepskyblue', alpha=0.5, label="_nolegend_")#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.plot(λ_fit, rdata1.lossy_model(rdata1.λ_fit, *rparams1), 'lightcoral', alpha=0.4)#, label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams1))
ax.scatter(tdata[:,0], tdata[:,1], marker=".", color="orangered")#, label='$T_{G2}$')
ax.plot(λ_fit, tfit, 'orangered', alpha=0.5, label = "_nolegend_")#, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$nm^{-1}$' % tuple(tparams))
#plt.plot(λ_fit, #lossy_model(tdata1.λ_fit, *tparams1), 'skyblue', alpha=0.4, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$m^{-1}$' % tuple(tparams1))

ax.scatter(rdata2[:,0], rdata2[:,1], marker=".", color="darkblue")#, label='$R_{G2}$')
ax.plot(λ_fit, rfit2, 'darkblue', alpha=0.5, label="_nolegend_")#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.plot(λ_fit, rdata1.lossy_model(rdata1.λ_fit, *rparams1), 'lightcoral', alpha=0.4)#, label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams1))
ax.scatter(tdata2[:,0], tdata2[:,1], marker=".", color="darkred", label="_nolegend_")#, label='$T_{G2}$')
ax.plot(λ_fit, tfit2, 'darkred', alpha=0.5, label = "_nolegend_")#, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$nm^{-1}$' % tuple(tparams))
#plt.plot(rdata2.λ_fit, rdata2.lossy_model(rdata2.λ_fit, *rparams2), 'darkred', alpha=0.4)#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.plot(rdata2.λ_fit, rdata2.lossy_model(rdata2.λ_fit, *rparams2), 'darkred', alpha=0.4)#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(rparams))
#plt.scatter(tdata2.data[:,0], tdata2.data[:,1], marker=".", color="darkblue")#, label='$T_{G2}$')
#plt.plot(tdata2.λ_fit, tdata2.lossy_model(tdata2.λ_fit, *tparams2), 'darkblue', alpha=0.4, label='$\\lambda_0 = $%5.3fnm, $\\lambda_1 = $%5.3fnm, $t_d = $%5.3f, $\\gamma_{\\lambda} = $%5.3fnm, $\\beta = $%.0e$m^{-1}$' % tuple(tparams2))
#plt.plot(tdata2.λ_fit, tdata2.lossy_model(tdata2.λ_fit, *tparams2), 'darkblue', alpha=0.4, label="$\\lambda_0^{\\prime} = $ %snm" % str(round(tparams2[0],2)))#label='fit: λ0=%5.3f, λ1=%5.3f, td=%5.3f, γ=%5.3f, α=%5.3f' % tuple(tparams))
ax.set_xlabel("Wavelength [nm]", fontsize=36)
#ax.set_ylabel("Grating transmission", fontsize=36, color="firebrick")
#ax.set_ylabel("Grating reflectivity", fontsize=36, color="royalblue")
ax.tick_params(axis='y', labelsize=28, direction="in", length=4, width=1)
ax.tick_params(axis='x', labelsize=28, direction="in", length=4, width=1)
#ax.xaxis.set_ticks_position("both")
#ax.yaxis.set_ticks_position("both")
#ax2.set_yticks([])
#ax2.tick_params(axis='both', direction="in")
#plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)
#plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
#plt.subplots_adjust(bottom=0.3)
plt.locator_params(axis='x', tight=True, nbins=8)
ax.set_ylim((0,1))
#ax2.set_ylim((0,1))
ax.grid(alpha=0.3)
plt.subplots_adjust(bottom=0.15)

idx1 = np.argmin(tfit)
idx2 = np.argmin(tfit2)

print("L (M3):", 1-rfit[idx1]-tfit[idx1], "+/-", np.abs((1-rfit_p[idx1]-tfit_p[idx1]) - (1-rfit_m[idx1]-tfit_m[idx1])))
print("L (M5):", 1-rfit2[idx2]-tfit2[idx2], "+/-", np.abs((1-rfit_p[idx2]-tfit_p[idx2]) - (1-rfit_m[idx2]-tfit_m[idx2])))
print("M3: rmax = ", max(rdata[:,1]), "tmin = ", min(tdata[:,1]))
print("M5: rmax = ", max(rdata2[:,1]), "tmin = ", min(tdata2[:,1]))

# --- Create custom legend handles showing two different colors (solid) ---
from matplotlib.lines import Line2D

def double_color_handle(color1, color2):
    """Create a legend handle that shows two solid lines of different colors."""
    return (  # return as tuple so handler_map recognizes it
        Line2D([], [], color=color1, linestyle='-', linewidth=2),
        Line2D([], [], color=color2, linestyle='-', linewidth=2)
    )

transmission_handle = double_color_handle('deepskyblue', 'darkblue')
reflection_handle   = double_color_handle('orangered', 'darkred')

class DoubleLineHandler:
    """Custom legend handler that draws two lines in one legend entry."""
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        l1, l2 = orig_handle
        y_center = handlebox.height / 2
        # draw first line (upper)
        handlebox.add_artist(Line2D(
            [0, handlebox.width], [y_center + 4, y_center + 4],
            color=l1.get_color(), linestyle=l1.get_linestyle(),
            linewidth=l1.get_linewidth(), transform=handlebox.get_transform()))
        # draw second line (lower)
        handlebox.add_artist(Line2D(
            [0, handlebox.width], [y_center - 4, y_center - 4],
            color=l2.get_color(), linestyle=l2.get_linestyle(),
            linewidth=l2.get_linewidth(), transform=handlebox.get_transform()))
        return None

# Add legend using our custom handler
ax.legend(
    [transmission_handle, reflection_handle],
    [r'$|r_g|^2$', r'$|t_g|^2$'],
    handler_map={tuple: DoubleLineHandler()},
    #loc='upper center',
    fontsize=28,
    #bbox_to_anchor=(0.5, -0.15),
    #ncol=2,
    #fancybox=True,
    #shadow=True
)

plt.show()

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






    



