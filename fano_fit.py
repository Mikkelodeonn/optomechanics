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


### Calculating the simulated reflection values 

def theoretical_reflection_values(params: list, λs: np.array, losses=True, loss_factor=0.03):
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

### Load data from .txt file

left = 0 #7
right = -1 #-6
extrapolated = False
line_width_fit = True

cavity_length_guess = 77#139#31

scan_num = 4
### fit FSR scans for all lengths from 20250326 -> plot in HWHM vs. cavity length figure
scan_type = "s"

#data = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250523/30um/"+str(scan_type)+str(scan_num)+".txt")#[left:right]
#PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250523/30um/"+str(scan_type)+str(scan_num)+"_PI.txt")#[left:right]
#norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250523/normalization/short_scan.txt")#[left:right]
#norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250523/normalization/short_scan_PI.txt")#[left:right]

data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/30um/"+str(scan_type)+str(scan_num)+".txt")#[left:right]
PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/30um/"+str(scan_type)+str(scan_num)+"_PI.txt")#[left:right]
#data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250211/34um/34l.txt")#[left:right]
#PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250211/34um/34l_PI.txt")#[left:right]
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/normalization/short_scan.txt")#[left:right]
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/normalization/short_scan_PI.txt")#[left:right]

if not np.allclose(data[:,0], norm[:,0]):
    raise Exception("Normalization and data files do not match!")

#PI_0 = PI_data[0,1]
#PI_0 = norm[0,1]
#data[:,1] = [dat/(PI/PI_0) for dat,PI in zip(data[:,1], norm_PI[:,1])]  ##  correcting for any fluctuations in laser power over time
#norm[:,1] = [N/(PI/PI_0_norm) for N,PI in zip(norm[:,1], norm_PI[:,1])]      ##  from the main measurement to the normalization measurement.  

data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
#output = np.column_stack((data[:,0], data[:,1]))

#np.savetxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250422/normalized data/500um/" + str(scan_type) + str(scan_num) + "_normalized.txt", output, fmt="%.6e", comments='')



### Calculate the parameters used for fitting the fano transmission function 

M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")



λ0_1, λ1_1, td_1, γ_1, α_1 = M5.lossy_fit([952,952,0.6,1,0.1])
λ0_2, λ1_2, td_2, γ_2, α_2 = M3.lossy_fit([952,952,0.6,1,0.1])
#M3 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M3_trans.txt")
#M5 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M5_trans.txt")
#M3_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M3_trans_PI.txt")
#M5_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/grating trans. spectra/M5_trans_PI.txt")
#M3_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans.txt")
#M5_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans.txt")
#M3_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans_PI.txt")
#M5_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250311/normalization/grating_trans_PI.txt")

#M3[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M3[:,1], M3_PI[:,1], M3_norm[:,1], M3_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
#M5[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5[:,1], M5_PI[:,1], M5_norm[:,1], M5_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

#λs = np.linspace(M3[:,0][0], M3[:,0][-1], 50)
#λs_fit = np.linspace(M3[:,0][0], M3[:,0][-1], 10000)

#p0 = [951.83,951.83,0.6,1,0.1]
#params1, pcov1 = curve_fit(model, M3[:,0], M3[:,1], p0=p0)
#params2, pcov2 = curve_fit(model, M5[:,0], M5[:,1], p0=p0)

#λ0_1, λ1_1, td_1, γ_1, α_1 = params1
#λ0_2, λ1_2, td_2, γ_2, α_2 = params2

#λ0_1 = 951.630
#λ1_1 = 951.630 + 0.14

#λ0_2 = 951.870
#λ1_2 = 951.870 + 0.15

### Defining the grating transmission function/model

def fit_model(λ, a, b, c, λ0, δλ): 
    γ = 1 - c*((λ-λ0)/δλ)
    t = (a/(1 + ((λ-λ0)/(δλ*γ))**2)) + b
    return t

### Double Fano fitting function

def double_fano(λs, λ0_1, λ1_1, λ0_2, λ1_2, length, loss_factor): ## params -> [λ0, λ1, td, γ, α]
    params1 = [λ0_1, λ1_1, td_1, γ_1, α_1]
    params2 = [λ0_2, λ1_2, td_2, γ_2, α_2]
    #print("params1:",params1)
    #print("params2:",params2)
    
    reflection_values1 = theoretical_reflection_values(params1, λs, losses=True, loss_factor=loss_factor)[1]
    transmission_values1 = np.sqrt(model(λs, *params1))
    reflection_values2 = theoretical_reflection_values(params2, λs, losses=True, loss_factor=loss_factor)[1]
    transmission_values2 = np.sqrt(model(λs, *params2))

    def cavity_transmission(λ, rg1, tg1, rg2, tg2, length):
        T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*length)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length)))**2
        #T = np.abs(1/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length)))**2
        return T 

    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values1[i], transmission_values1[i], reflection_values2[i], transmission_values2[i], length)
        Ts.append(float(T))

    return Ts

### Fitting loaded data to the double fano transmission function

if line_width_fit == False:
    p0 = [λ0_1, λ1_1, λ0_2, λ1_2, cavity_length_guess*1e3, 0.2]
    bounds = ([0,0,0,0,0,0,0,0,0,0,0,0],[np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, 1, np.inf, 1, np.inf, 1])
    popt,pcov = curve_fit(double_fano, data[:,0], data[:,1], p0=p0, maxfev=10000000)
    errs = np.sqrt(np.diag(pcov))
    #fit_params = [popt[0], popt[1], popt[5], popt[6], popt[10]*1e-3]

    xs = np.linspace(data[:,0][0], data[:,0][-1], 10000) 

    residuals = data[:,1] - double_fano(data[:,0], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data[:,1]-np.mean(data[:,1]))**2)
    r_squared = 1 - (ss_res/ss_tot)
    print("r^2 = ", r_squared)

    plt.figure(figsize=(10,7))
    plt.scatter(data[:,0], data[:,1], color="maroon", marker=".", label="data")
    plt.plot(xs, double_fano(xs, *popt), color="orangered", alpha=0.7, label="fit")
    #print("G1: \nλ0 = ",popt[0], "+/-", errs[0], "nm", "\nλ1 = ",popt[1], "+/-", errs[1], "nm", "\ntd = ", popt[2], "+/-", errs[2], "\nγλ = ", popt[3], "+/-", errs[3], "nm", "\nα = ", popt[4], "+/-", errs[4])  
    #print("G2: \nλ0 = ",popt[5], "+/-", errs[5], "nm", "\nλ1 = ",popt[6], "+/-", errs[6], "nm", "\ntd = ", popt[7], "+/-", errs[7], "\nγλ = ", popt[8], "+/-", errs[8], "nm", "\nα = ", popt[9], "+/-", errs[9])
    print("cavity length: ", popt[4]*1e-3, "+/-", errs[4]*1e-3, "μm") 
    print("losses: ", popt[5]*2, "+/-", errs[5]*2)
    print("λ0_G1 ", popt[0], "+/-", errs[0],"  λ1_G1: ", popt[1], "+/-", errs[1])
    print("λ0_G2 ", popt[2], "+/-", errs[2],"  λ1_G2: ", popt[3], "+/-", errs[3])
    #plt.title("M3/M5 double fano transmission")  
    plt.xlabel("wavelength [nm]", fontsize=28)
    plt.ylabel("norm. trans.", fontsize=28)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    #plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
    plt.subplots_adjust(bottom=0.3)
    #plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.show()
else:
    p0 = [1, 0.1, 0, 951.900, 30e-3]
    bounds = [[0, 0, -np.inf, 0, 0],[np.inf, np.inf, np.inf, np.inf, np.inf]]
    popt,pcov = curve_fit(fit_model, data[:,0], data[:,1], p0=p0, bounds=bounds, maxfev=1000000)

    lw_err = round(np.sqrt(np.diag(pcov))[4]*1e3,3)
    hwhm = round(np.abs(popt[4])*1e3,3)
    legend = [hwhm, lw_err]
    print("linewidth: ", hwhm, "+/-", lw_err)
    #print("popt:",popt)
    #print("p0 =", p0)

    if extrapolated == False:
        xs = np.linspace(data[:,0][0], data[:,0][-1], 10000) 
    else:
        xs = np.linspace(data[:,0][0]-1, data[:,0][-1]+1, 10000) 

    plt.figure(figsize=(11,7))
    plt.scatter(data[:,0], data[:,1], color="magenta", marker="o", label="data", zorder=1)
    plt.plot(xs, fit_model(xs, *popt), color="magenta", alpha=0.7, label="fit: HWHM $\\approx$ %5.3f +/- %5.3fpm" % tuple(legend))
    #plt.title("M3/M5 double fano transmission")  
    plt.xlabel("Wavelength [nm]", fontsize=36)
    plt.ylabel("Cavity transmission", fontsize=36)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    #plt.subplots_adjust(bottom=0.2)
    #plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.locator_params(axis='x', tight=True, nbins=5)
    #plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid(alpha=0.3)
    #plt.show()