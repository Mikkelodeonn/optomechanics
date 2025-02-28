from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

M1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M1/400_M1 trans.txt")
M2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M2/400_M2 trans.txt")
M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M4 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M4/400_M4 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")
M7 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M7/400_M7 trans.txt")

params1 = M3.lossy_fit([952,952,0.6,1,0.1])
params2 = M5.lossy_fit([952,952,0.6,1,0.1])

#params3 = M7.lossy_fit([952,952,0.6,1,0.1])

#λ_asymmetry_1 = params1[1]-params1[0]
#λ_asymmetry_2 = params2[1]-params2[0]

params1[0] = 951.635833090383
params1[1] = 951.7808553381275
#params1[2] = 0.8112460470002542 
#params1[3] = 0.5127336548338813
#params1[4] = 9.089548008029833e-07 

###951.630 + λ_asymmetry_1

params2[0] = 952.0676067923679
params2[1] = 952.1863127407705
#params2[2] = 0.8206794339330804
#params2[3] = 0.6375760748025334
#params2[4] = 2.4859509672570234e-06

### 951.870 + λ_asymmetry_2

## grating parameters -> [λ0, λ1, td, γλ, α]
# λ0 -> resonance wavelengths
# λ1 -> guided mode resonance wavelength
# td -> direct transmission coefficient
# γλ -> width of guided mode resonance
# α  -> loss factor 

data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/20um/20s5.txt")
PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/20um/20s5_PI.txt")
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/normalization/short_scan.txt")
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/normalization/short_scan_PI.txt")
PI_0 = PI_data[0,1]
data[:,1] = [dat/(PI/PI_0) for dat,PI in zip(data[:,1], PI_data[:,1])]
norm[:,1] = [N/(PI/PI_0) for N,PI in zip(norm[:,1], norm_PI[:,1])]

data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])]

#λs = np.linspace(951, 952.5, 500)
#λs = np.linspace(951.65, 951.95, 500)
#λs = np.linspace(M3.data[:,0][0], M3.data[:,0][-1], 100)
#λs = np.linspace(data[:,0][0], data[:,0][-1], 1000)
#λs = np.linspace(951.650, 951.950, 100)
λs = np.linspace(951.500, 951.800, 50)
#λs = np.linspace(910, 980, 10000)
#λs = np.linspace(951.68, 951.90, 200)

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def fit_model(λ, a, b, c, λ0, δλ): 
    γ = 1 - c*((λ-λ0)/δλ)
    t = (a/(1 + ((λ-λ0)/(δλ*γ))**2)) + b
    return t

def theoretical_reflection_values(params: list, losses=True, loss_factor=0.05):
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

def theoretical_phase_values(params: list, losses=True, loss_factor=0.03):
    rs_complex = np.array(theoretical_reflection_values(params, losses=losses, loss_factor=loss_factor)[1])
    φs = np.angle(rs_complex)
    return φs

def theoretical_phase_plot(params: list, λs: np.array, losses=True, loss_factor=0.05):
    plt.figure(figsize=(10,7))
    φs = theoretical_phase_values(params, losses=losses, loss_factor=loss_factor)
    plt.scatter(λs, φs, marker=".", color="royalblue", label="simulated phase")
    plt.plot(λs, φs, color="cornflowerblue")
    plt.title("Simulated phase as a function of wavelength (M3)")
    plt.ylabel("φ(λ) [radians]")
    plt.xlabel("wavelength [nm]")
    plt.legend()
    plt.show()

def theoretical_reflection_values_plot(params: list, λs: np.array):
    plt.figure(figsize=(10,7))
    rs = theoretical_reflection_values(params, losses=True)[0]
    rs = [float(r) for r in rs]
    ts = model(λs, *params)
    λs_fit = np.linspace(np.min(λs), np.max(λs), 1000)

    popt_r, _ = curve_fit(model, λs, rs, p0=[951.8,951.8,0.4,1,1e-7])
    popt_t, _ = curve_fit(model, λs, ts, p0=[951.8,951.8,0.6,1,1e-7])

    ts_fit = model(λs_fit, *popt_t)
    rs_fit = model(λs_fit, *popt_r)

    tidx = np.argmin(ts_fit)
    ridx = np.argmax(rs_fit)

    rmax = rs_fit[ridx]
    tmin = ts_fit[tidx]

    plt.title("Simulated transmission/reflection values")
    plt.plot(λs, rs, 'ro', label="simulated reflection values")
    plt.plot(λs, ts, 'bo', label="simulated transmission data")
    plt.plot(λs_fit, ts_fit, 'darkblue', label="minimum transmission: %s %%" % str(round(tmin*1e2,2)))
    plt.plot(λs_fit, rs_fit, 'darkred', label="maximum reflectivity: %s %%" % str(round(rmax*1e2,2)))
    plt.ylabel("normalized transmission/reflection [arb. u.]")
    plt.xlabel("wavelength [nm]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    plt.show()

def theoretical_reflection_values_comparison_plot(params1: list, params2: list, λs: np.array):
    plt.figure(figsize=(15,6))
    r1 = theoretical_reflection_values(params1, losses=True)[0]
    r1 = [float(r) for r in r1]
    r2 = theoretical_reflection_values(params2, losses=True)[0]
    r2 = [float(r) for r in r2]

    t1 = model(λs, *params1)
    t2 = model(λs, *params2)
    
    λs_fit = np.linspace(np.min(λs), np.max(λs), 1000)

    popt_t1, _ = curve_fit(model, λs, t1, p0=[952,952,0.6,1,1e-7])
    popt_t2, _ = curve_fit(model, λs, t2, p0=[952,952,0.6,1,1e-7])
    t1_fit = model(λs_fit, *popt_t1)
    t2_fit = model(λs_fit, *popt_t2)
    tidx1 = np.argmin(t1_fit)
    tidx2 = np.argmin(t2_fit)
    tmin1 = t1_fit[tidx1]
    tmin2 = t2_fit[tidx2]

    popt_r1, _ = curve_fit(model, λs, r1, p0=[952,952,0.6,1,1e-7])
    popt_r2, _ = curve_fit(model, λs, r2, p0=[952,952,0.6,1,1e-7])
    r1_fit = model(λs_fit, *popt_r1)
    r2_fit = model(λs_fit, *popt_r2)
    ridx1 = np.argmax(r1_fit)
    ridx2 = np.argmax(r2_fit)
    rmax1 = r1_fit[ridx1]
    rmax2 = r2_fit[ridx2]

    plt.title("Simulated transmission/reflection values")
    plt.plot(λs, r1, 'o', color="darkred", label="ref. M3 (%snm)" % str("$\\lambda_{0} = $" + str(round(popt_r1[0],2))))
    plt.plot(λs, t1, 'o',  color="darkblue", label="trans. M3 (%snm)" % str("$\\lambda_{0} = $" + str(round(popt_t1[0],2))))
    plt.plot(λs, r2, 'o', color="firebrick", alpha=0.6, label="ref. M5 (%snm)" % str("$\\lambda_{0} = $" + str(round(popt_r2[0],2))))
    plt.plot(λs, t2, 'o', color="royalblue", alpha=0.6, label="trans. M5 (%snm)" % str("$\\lambda_{0} = $" + str(round(popt_t2[0],2))))
    plt.plot(λs_fit, r1_fit, 'darkred', label="$r_{max,M3}$: %s %%" % str(round(rmax1*1e2,2)))
    plt.plot(λs_fit, t1_fit, 'darkblue', label="$t_{min,M3}$: %s %%" % str(round(tmin1*1e2,2)))
    plt.plot(λs_fit, r2_fit, 'firebrick', alpha=0.6, label="$r_{max,M5}$: %s %%" % str(round(rmax2*1e2,2)))
    plt.plot(λs_fit, t2_fit, 'royalblue', alpha=0.6, label="$t_{min,M5}$: %s %%" % str(round(tmin2*1e2,2)))
    plt.ylabel("normalized transmission/reflection [arb. u.]")
    plt.xlabel("wavelength [nm]")
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.70)
    plt.show()

def resonant_cavity_length(params: list, λs: np.array, lmin=50):
    reflection_values = theoretical_reflection_values(params, losses=True)[1]
    transmission_values = np.sqrt(model(λs, *params))

    reflection_values = [complex(r) for r in reflection_values]
    transmission_values = [complex(t) for t in transmission_values]

    idx = np.argmin(transmission_values)

    lengths = []
    Ts = []

    tg = transmission_values[idx]
    rg = reflection_values[idx]
    tm = np.sqrt(0.08)
    rm = np.sqrt(0.92)

    ls = list(np.linspace(lmin,lmin+1,100000)*1e3)

    for l in ls:
        λ = λs[idx]
        t = np.abs(tg*tm*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
        Ts.append(t)

    peak_indices = find_peaks(Ts)

    for idx in peak_indices[0]:
        lengths.append(ls[idx])

    if np.abs(Ts[ls.index(lengths[0])] - Ts[ls.index(lengths[1])]) > 1e10:
        resonance_length = lengths[1]
    else:
        resonance_length = lengths[0]

    return resonance_length 

def double_cavity_length(params1: list, params2: list, λs: np.array, lmin=50, loss_factor=0.05):
    r1 = theoretical_reflection_values(params1, losses=True, loss_factor=loss_factor)[1]
    r2 = theoretical_reflection_values(params2, losses=True, loss_factor=loss_factor)[1]
    t1 = np.sqrt(model(λs, *params1))
    t2 = np.sqrt(model(λs, *params2))

    r1 = [complex(r) for r in r1]; r2 = [complex(r) for r in r2]
    t1 = [complex(t) for t in t1]; t2 = [complex(t) for t in t2]

    idx = np.argmin(np.array(t1))

    rg1 = r1[idx]; rg2 = r2[idx]
    tg1 = t1[idx]; tg2 = t2[idx]

    lengths = []
    Ts = []

    ls = list(np.linspace(lmin,lmin+1,100000)*1e3)

    for l in ls:
        λ = λs[idx]
        t = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
        Ts.append(t)

    peak_indices = find_peaks(Ts)

    for idx in peak_indices[0]:
        lengths.append(ls[idx])

    if np.abs(Ts[ls.index(lengths[0])] - Ts[ls.index(lengths[1])]) > 1e10:
        resonance_length = lengths[1]
    else:
        resonance_length = lengths[0]

    return resonance_length 

def single_fano_length_scan(params: list, ls: np.array, λs: np.array):
    ts = [complex(t) for t in np.array(np.sqrt(model(λs, *params)))]
    rs = [complex(r) for r in theoretical_reflection_values(params, losses=True)[1]]

    idx = np.argmin(ts)
    rg = rs[idx]
    tg = ts[idx]
    rm = np.sqrt(0.92)
    tm = np.sqrt(0.08)

    λ = np.array([λs[idx]])
    Ts = []
    for l in ls:
        T = np.abs(tg*tm*np.exp(1j*(2*np.pi/λ)*l)/(1-rg*rm*np.exp(2j*(2*np.pi/λ)*l)))**2
        Ts.append(T)

    resonance_length = resonant_cavity_length(params, λ, lmin=ls[0]*1e-3)

    plt.figure(figsize=(10,6))
    plt.title("Fano cavity transmission as a function of cavity length")
    plt.plot(ls*1e-3,Ts, "cornflowerblue")
    plt.xlabel("cavity length [μm]")
    plt.ylabel("Transmission [arb. u.]")
    plt.legend(["Resonance cavity length: %sμm" % str(round(resonance_length*1e-3,3))])
    plt.show()

def double_fano_length_scan(params1: list, params2: list, ls: np.array, λs: np.array, plot_both_gratings: False):
    t1 = [complex(t) for t in np.array(np.sqrt(model(λs, *params1)))]
    t2 = [complex(t) for t in np.array(np.sqrt(model(λs, *params2)))]
    r1 = [complex(r) for r in theoretical_reflection_values(params1, losses=True)[1]]
    r2 = [complex(r) for r in theoretical_reflection_values(params2, losses=True)[1]]

    idx = np.argmin(t1)
    idx_2 = np.argmin(t2)

    rg1 = r1[idx]; rg2 = r2[idx]
    tg1 = t1[idx]; tg2 = t2[idx]

    rg1_2 = r1[idx_2]; rg2_2 = r2[idx_2]
    tg1_2 = t1[idx_2]; tg2_2 = t2[idx_2]


    λ = np.array([λs[idx]])
    λ_2 = np.array([λs[idx_2]])
    Ts = []
    Ts_2 = []
    for l in ls:
        T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
        T_2 = np.abs(tg1_2*tg2_2*np.exp(1j*(2*np.pi/λ_2)*l)/(1-rg1_2*rg2_2*np.exp(2j*(2*np.pi/λ_2)*l)))**2
        Ts.append(T)
        Ts_2.append(T_2)
    
    resonance_length = double_cavity_length(params1, params2, λs, lmin=ls[0]*1e-3)*1e-3
    resonance_length_2 = double_cavity_length(params2, params1, λs, lmin=ls[0]*1e-3)*1e-3

    plt.figure(figsize=(15,6))
    if plot_both_gratings == True:
        plt.plot(ls*1e-3, Ts, "cornflowerblue", label="$l_{res,M3}$: %sμm" % str(round(resonance_length,3)))
        plt.plot(ls*1e-3, Ts_2, "--", color="lightcoral", label="$l_{res,M5}$: %sμm" % str(round(resonance_length_2,3)))
    else:
        plt.plot(ls*1e-3, Ts, "cornflowerblue", label="$l_{res,M3}$: %sμm" % str(round(resonance_length,3)))
    plt.title("Double fano cavity transmission as a function of cavity length")
    plt.xlabel("cavity length [μm]")
    plt.ylabel("Transmission [arb. u.]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.70)
    plt.show()
 
def fano_cavity_transmission(params: list, length: np.array, λs: np.array, intracavity=False, losses=True):
    #print("single fano length:", length)

    reflection_values = theoretical_reflection_values(params, losses=losses)[1]
    transmission_values = np.sqrt(model(λs, *params))

    if intracavity == False:
        def cavity_transmission(λ, rg, tg, l):
            tm = np.sqrt(0.08)
            rm = np.sqrt(0.92)
            T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 
        
    if intracavity == True:
        def cavity_transmission(λ, rg, tg, l):
            rm = np.sqrt(0.92)
            tg = 1
            tm = 1
            T = np.abs(tg*tm*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 
    
    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values[i], transmission_values[i], length)
        Ts.append(float(T))

    return Ts

def single_fano_phase(params: list, length: np.array, λs: np.array, intracavity=False, losses=True):

    reflection_values = theoretical_reflection_values(params, losses=losses)[1]
    transmission_values = np.sqrt(model(λs, *params))

    if intracavity == False:
        def cavity_transmission(λ, rg, tg, l):
            tm = np.sqrt(0.08)
            rm = np.sqrt(0.92)
            T = tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l))
            return T 
        
    if intracavity == True:
        def cavity_transmission(λ, rg, tg, l):
            rm = np.sqrt(0.92)
            tg = 1
            tm = 1
            T = tg*tm*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l))
            return T 
    
    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values[i], transmission_values[i], length)
        Ts.append(T)

    φs = np.angle(Ts)
    for φ in φs:
        if φ < 0:
            φ += 2*np.pi
    return φs

def single_fano_phase_plot(params: list, length: np.array, λs: np.array, intracavity=False, losses=True):
    plt.figure(figsize=(10,7))
    φs = single_fano_phase(params, length, λs, intracavity=intracavity, losses=losses)
    plt.scatter(λs, φs, marker=".", color="royalblue", label="simulated phase")
    plt.plot(λs, φs, color="cornflowerblue")
    plt.title("Single Fano: phase as a function of wavelength")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("φ(λ) [radians]")
    plt.legend()
    plt.show()

def fano_cavity_transmission_plot(params: list, length: np.array, λs: np.array, intracavity=False, losses=True):
    Ts = fano_cavity_transmission(params, length, λs, intracavity=intracavity, losses=losses)
    plt.figure(figsize=(10,6))
    plt.plot(λs, Ts, color="royalblue", label="simulated phase")
    plt.title("Single fano cavity transmission as function of wavelength (l = %sμm)" % str(round(length*1e-3,2)))
    plt.xlabel("Wavelength [nm]") 
    plt.ylabel("Intensity [arb.u.]")
    plt.show()

def dual_fano_transmission(params1: list, params2: list, length: float, λs: np.array, intracavity=False, losses=True, loss_factor=0.05):
    #print("double fano length: ", length)
    
    reflection_values1 = theoretical_reflection_values(params1, losses=losses, loss_factor=loss_factor)[1]
    transmission_values1 = np.sqrt(model(λs, *params1))
    reflection_values2 = theoretical_reflection_values(params2, losses=losses, loss_factor=loss_factor)[1]
    transmission_values2 = np.sqrt(model(λs, *params2))

    if intracavity == False:
        def cavity_transmission(λ, rg1, tg1, rg2, tg2, length):
            T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*length)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length)))**2
            return T 
        
    if intracavity == True:
        def cavity_transmission(λ, rg1, tg1, rg2, tg2, length):
            tg1 = 1; tg2 = 1
            T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*length)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length)))**2
            return T 

    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values1[i], transmission_values1[i], reflection_values2[i], transmission_values2[i], length)
        Ts.append(float(T))

    return Ts

def double_fano_phase(params1: list, params2: list, length: float, λs: np.array, intracavity=False, losses=True, loss_factor=0.03):
    reflection_values1 = theoretical_reflection_values(params1, losses=losses, loss_factor=loss_factor)[1]
    transmission_values1 = np.sqrt(model(λs, *params1))
    reflection_values2 = theoretical_reflection_values(params2, losses=losses, loss_factor=loss_factor)[1]
    transmission_values2 = np.sqrt(model(λs, *params2))

    if intracavity == False:
        def cavity_transmission(λ, rg1, tg1, rg2, tg2, length):
            T = tg1*tg2*np.exp(1j*(2*np.pi/λ)*length)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length))
            return T 
        
    if intracavity == True:
        def cavity_transmission(λ, rg1, tg1, rg2, tg2, length):
            tg1 = 1; tg2 = 1
            T = tg1*tg2*np.exp(1j*(2*np.pi/λ)*length)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length))
            return T 

    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values1[i], transmission_values1[i], reflection_values2[i], transmission_values2[i], length)
        Ts.append(T)

    φs = np.angle(Ts)
    for φ in φs:
        if φ < 0:
            φ += 2*np.pi
    return φs

def double_fano_phase_plot(params1: list, params2: list, length: float, λs: np.array, intracavity=False, losses=True, loss_factor=0.05):
    φs = double_fano_phase(params1, params2, length, λs, intracavity=intracavity, losses=losses, loss_factor=loss_factor)
    plt.figure(figsize=(10,7))
    plt.scatter(λs, φs, marker=".", color="royalblue", label="simulated phase")
    plt.plot(λs, φs, color="cornflowerblue")
    plt.title("double fano: phase as a function of wavelength")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("φ(λ) [radians]")
    plt.legend()
    plt.show()

def dual_fano_transmission_plot(params1: list, params2: list, length: float, λs: np.array, intracavity=False, losses=True, zoom=False, grating_trans=False):
    Ts =  dual_fano_transmission(params1, params2, length, λs, intracavity=intracavity, losses=losses)
    fig, ax = plt.subplots(figsize=(10,6))
    if zoom == True and grating_trans == False:
        x1, x2, y1, y2 = params1[0]-1, params1[0]+1, 0.05, 0.6
        axins = ax.inset_axes([0.1, 0.50, 0.30, 0.30])
        axins.plot(λs,Ts)
        axins.set_xlim(x1,x2)
        axins.set_ylim(y1,y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="black", alpha=0.3)
    if grating_trans == True and zoom == False:
        tg1 = model(λs, *params1)
        tg2 = model(λs, *params2)
        ax.plot(λs, tg1, "gray", linestyle="--", alpha=0.6, label="$t_{M3}$")
        ax.plot(λs, tg2, "purple", linestyle="--", alpha=0.6, label="$t_{M5}$")
    if grating_trans == True and zoom == True:
        tg1 = model(λs, *params1)
        tg2 = model(λs, *params2)
        ax.plot(λs, tg1, "gray", linestyle="--", alpha=0.6, label="$t_{M3}$")
        ax.plot(λs, tg2, "purple", linestyle="--", alpha=0.6, label="$t_{M5}$")
        x1, x2, y1, y2 = params1[0]-1, params1[0]+1, 0.01, 0.6
        axins = ax.inset_axes([0.1, 0.50, 0.30, 0.30])
        axins.plot(λs, Ts, color="cornflowerblue")
        axins.plot(λs, tg1, "gray", linestyle="--", alpha=0.6)
        axins.plot(λs, tg2, "purple", linestyle="--", alpha=0.6)
        axins.set_xlim(x1,x2)
        axins.set_ylim(y1,y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="black", alpha=0.3)
    ax.set_title("Double fano transmission as a function of wavelength ($\\left(l_{M3} + l_{M5}\\right)/2 \\approx$%sμm)" % str(round(length*1e-3,3)))
    #ax.set_title("Double fano transmission as a function of wavelength ($l_{M3} \\approx$%sμm)" % str(round(length*1e-3,3)))
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Intensity [arb.u.]")
    ax.plot(λs, Ts, "cornflowerblue", label="cavity transmission")
    ax.legend()
    plt.show()

def detuning_plot(Δs: list, params: list, λs: np.array, intracavity=False, losses=True, lmin=50): ## plots dual fano cavity transmission for different values for the detuning
    plt.figure(figsize=(10,6))
    #length = double_cavity_length(params, params, λs, lmin=lmin)
    linestyles = ["-.", "--", "-", "--", "-."]
    colors = ["skyblue","royalblue","forestgreen", "firebrick", "lightcoral"]
    for Δ, paint, style in zip(Δs,colors,linestyles):
        params2 = np.copy(params)
        params2[0] += Δ
        params2[1] += Δ
        length = (double_cavity_length(params, params2, λs, lmin=lmin) + double_cavity_length(params2, params, λs, lmin=lmin))/2
        Ts =  dual_fano_transmission(params, params2, length, λs, intracavity=intracavity, losses=losses)
        if np.abs(Δ) < 1e-6:
            linesize = 3
        else:
            linesize = 2
        plt.plot(λs, Ts, color=paint, linestyle=style, linewidth=linesize, label="Δ=%snm" %(round(Δ,2)))

    #plt.title("Double fano transmission for varying detuning Δ (cavity length %s)" %(r"$\rightarrow l_{g,1} \approx 100 \mu m$"))
    plt.title("Double fano transmission for varying detuning Δ (cavity length %s)" %(r"$\rightarrow (l_{g,1} + l_{g,2})/2 \approx 10 \mu m$"))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [arb.u.]")
    plt.legend()
    plt.show()

def loss_factor_scan(params: list, loss_list: list, λs: np.array, lmin=50): ## only makes sense for symmetric double fano
    plt.figure(figsize=(15,6))
    linestyles = ["-.", "--", "-", "--", "-."]
    colors = ["skyblue","lightcoral","lightgreen", "firebrick", "forestgreen"]
    for loss, linestyle, color in zip(loss_list, linestyles, colors):
        r_max = np.max(theoretical_reflection_values(params, loss_factor=loss)[0])
        t_min = np.min(np.real(model(λs, *params)))
        percentile_loss = round((1-r_max-t_min)*1e2,2) ## in percent
        length = double_cavity_length(params, params, λs, lmin=lmin, loss_factor=loss)
        Ts = dual_fano_transmission(params1, params2, length, λs, loss_factor=loss)
        plt.plot(λs, Ts, color=color, linestyle=linestyle, linewidth=2, label="cavity losses: %s%%" %str(2*np.abs(percentile_loss)))
    plt.title("symmetric double fano transmission for different cavity losses, length: $l_{M3} \\approx$ %s $\mu m$" % str(lmin))
    plt.xlabel("wavelength [nm]")
    plt.ylabel("normalized transmission [arb. u.]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.70)
    plt.show()

def cavity_length_plot(ls: list, params1: list, params2: list, λs: np.array, intracavity=False, losses=True, zoom=False):
    fig, ax = plt.subplots(figsize=(15,6))
    linestyles = ["-.", "--", "-", "--", "-."]
    #linestyles = [":", "-.", "--", "-"]
    colors = ["skyblue","royalblue","forestgreen", "firebrick", "lightcoral"]
    #colors = ["skyblue", "lightgreen", "lightcoral", "royalblue"]
    Ts_inset = []
    for l, paint, style in zip(ls, colors, linestyles):
        Ts = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
        ax.plot(λs, Ts, color=paint, linestyle=style, linewidth=2, label="cavity length: %sμm" % str(round(l*1e-3,4)))
        Ts_inset.append(Ts)
    if zoom == True:
        axins = ax.inset_axes([0.1, 0.60, 0.35, 0.35])
        for Ts, paint, style in zip(Ts_inset,colors,linestyles):
            axins.plot(λs, Ts, color=paint, linestyle=style, linewidth=2)
        axins.set_xlim(951.65, 951.95)
        #axins.set_xlim(950, 953.5)
        axins.set_ylim(0.02, 0.5)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="black", alpha=0.3)
    #plt.title("Double fano transmission for different cavity lengths %s" %("$l_{M3} \\rightarrow l_{M5}$")) 
    plt.title("Double fano transmission for different cavity lengths:  %s" %("$(l_{M3} + l_{M5})/2$")) 
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized transmission [arb.u.]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.70)
    plt.show() 

def l_vs_λ_cmaps(params1: list, params2: list, λs: np.array, intracavity=False, losses=True, lmin=50): 
    params2[1] += 0.40
    params2[0] += 0.40
    Δs = 0.08
    rows = 3
    columns = 3
    Δ_label = 0.40
    fig, ax = plt.subplots(rows,columns, figsize=(18,8))
    for i in range(rows):
        for j in range(columns):
            params2[1] -= Δs
            params2[0] -= Δs
            Δ_label -= Δs
            if np.abs(Δ_label) < 1e-6:
                ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin)-0.1, double_cavity_length(params2, params1, λs, lmin=lmin)+0.1, 20)
            else:
                ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin), double_cavity_length(params2, params1, λs, lmin=lmin), 20)
            Ts = []
            for l in ls:
                T = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
                Ts.append(T)

            cmap = np.zeros([len(Ts),len(Ts[0])])

            for h in range(len(Ts)):
                for k in range(len(Ts[h])):
                    cmap[h,k] = Ts[h][k] 
            
            l_labels = [round(l*1e-3,2) for l in ls]
            λ_labels = np.linspace(np.min(λs), np.max(λs),10)
            λ_labels = [round(label,2) for label in λ_labels]

            im = ax[i,j].imshow(cmap, aspect="auto", extent=[np.min(λs), np.max(λs), np.min(ls), np.max(ls)])
            ax[i,j].set_title("Δ = %snm" %(round(Δ_label,2)), fontsize=7)
            ax[i,j].set_xticks(np.linspace(np.min(λs), np.max(λs),10))
            ax[i,j].set_xticklabels(λ_labels, fontsize=5)
            ax[i,j].set_yticks(ls)
            ax[i,j].set_yticklabels(l_labels, fontsize=5)    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if intracavity == False and losses == False:
        fig.text(0.5, 0.93, 'Double fano lossless transmission as a function of cavity length for different values of Δ', ha='center', va='center', fontsize=16) 
    elif intracavity == False and losses == True:
        fig.text(0.5, 0.93, 'Double fano transmission as a function of cavity length for different values of Δ', ha='center', va='center', fontsize=16) 
    else: 
        fig.text(0.5, 0.93, 'Double fano lossless intracavity intensity as a function of cavity length for different values of Δ', ha='center', va='center', fontsize=16) 
    fig.text(0.5, 0.06, 'Wavelength [nm]', ha='center', va='center', fontsize=10)
    fig.text(0.08, 0.5, 'Cavity length [μm]', ha='center', va='center', fontsize=10, rotation="vertical")
    plt.show()

def double_fano_cmap(params1: list, params2: list, λs: np.array, intracavity=False, losses=True, lmin=50):

    plt.figure(figsize=(10,6))

    ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin), double_cavity_length(params2, params1, λs, lmin=lmin),20)
    Ts = []
    for l in ls:
        T = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
        Ts.append(T)

    cmap = np.zeros([len(Ts),len(Ts[0])])

    for h in range(len(Ts)):
        for k in range(len(Ts[h])):
            cmap[h,k] = Ts[h][k] 

    l_labels = [round(l*1e-3,2) for l in ls]
    λ_labels = np.linspace(np.min(λs), np.max(λs),10)
    λ_labels = [round(label,2) for label in λ_labels]

    plt.imshow(cmap, aspect="auto", extent=[np.min(λs), np.max(λs), np.min(ls), np.max(ls)])
    plt.xticks(λ_labels)
    plt.yticks(ls, l_labels)
    ax = plt.gca()
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(axis='x', direction='inout', which='both')
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Cavity length [μm]")
    plt.colorbar()
    plt.show()

def line_width_double(params1: list, params2: list, λs: np.array, length: float, intracavity=False, losses=True): 
    Ts =  dual_fano_transmission(params1, params2, length, λs, intracavity=intracavity, losses=losses)

    popt, pcov = curve_fit(model, λs, Ts, p0=params1, maxfev=10000)

    FWHM = np.abs(2*popt[3])*1e3
    err = 2*np.sqrt(np.diag(pcov))[3]*1e3
    print("error: ", round(err,4))
    print("FWHM: ", round(np.abs(FWHM),4), "pm")
    FWHM_print = str(round(FWHM,2)) + " +/- " + str(round(err,2))
    plt.figure(figsize=(10,6))
    plt.plot(λs, model(λs, *popt), label="linewidth = %s" % (FWHM_print))
    plt.plot(λs, Ts, 'r.')
    plt.legend()
    plt.show()

    return FWHM*1e3

def line_width_single(params1: list, λs: np.array, intracavity=False, losses=True, lmin=50): 
    length = resonant_cavity_length(params1, λs, lmin=lmin)
    Ts =  fano_cavity_transmission(params1, length, λs, intracavity=intracavity, losses=losses)

    popt, pcov = curve_fit(model, λs, Ts, p0=params1, maxfev=10000)

    FWHM = np.abs(2*popt[3])*1e3

    plt.figure(figsize=(10,6))
    plt.plot(λs, model(λs, *popt), label="linewidth = %s" % str(FWHM))
    plt.plot(λs, Ts, 'r.')
    plt.legend()
    plt.show()

    return FWHM

def line_width_comparison(params1: list, params2: list, length: float, intracavity=False, losses=True): 
    T1 =  fano_cavity_transmission(params1, length, λs, intracavity=intracavity, losses=losses)
    T2 = dual_fano_transmission(params1, params2, length, λs, intracavity=intracavity, losses=losses)

    popt1, pcov1 = curve_fit(model, λs, T1, p0=params1, maxfev=10000)
    popt2, pcov2 = curve_fit(model, λs, T2, p0=params1, maxfev=10000)

    FWHM_single = np.abs(2*popt1[3])*1e3
    FWHM_double = np.abs(2*popt2[3])*1e3

    plt.figure(figsize=(10,6))
    plt.title("Double vs single fano comparison (M3 w/ losses)")
    plt.plot(λs, T1, '.', color="cornflowerblue", alpha=0.5, label="single fano simulation")
    plt.plot(λs, T2, 'g.', alpha=0.5, label="double fano simulation")
    plt.plot(λs, model(λs, *popt1), label="single fano fit, FWHM: %spm" %(str(round(FWHM_single,2))), color="orange")
    plt.plot(λs, model(λs, *popt2), label="double fano fit, FWHM: %spm" %(str(round(FWHM_double,2))))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb. u.]")
    plt.legend()
    plt.show()

    return FWHM

def linewidth_length_plot(params1: list, params2: list, λs: np.array, intracavity=True, losses=True):
    lws = []
    errs = []
    lengths = np.linspace(double_cavity_length(params1, params2, λs, lmin=30), double_cavity_length(params2, params1, λs, lmin=30), 10)

    for length in lengths:
        Ts =  dual_fano_transmission(params1, params2, length, λs, intracavity=intracavity, losses=losses)
        popt, pcov = curve_fit(model, λs, Ts, p0=(params1+params2)/2, maxfev=10000)
        FWHM = np.abs(2*popt[3])*1e3
        err = 2*np.sqrt(np.diag(pcov))[3]*1e3
        lws.append(FWHM)
        errs.append(err)
    
    plt.figure(figsize=(10,6))
    plt.errorbar(lengths*1e-3, lws, errs, fmt="o", color="cornflowerblue", capsize=5)
    plt.title("intracavity FWHM as a function of cavity length ($l = l_{M3} \\rightarrow l_{M5} \\approx 30μm$)")
    plt.ylabel("FWHM [pm]")
    plt.xlabel("cavity length [μm]") 
    plt.show()


#lmin = 10
#length = (double_cavity_length(params1, params2, λs, lmin=lmin) + double_cavity_length(params2, params1, λs, lmin=lmin))/2
#length = resonant_cavity_length(params1, λs, lmin)
#φs = double_fano_phase(params1, params2, length, λs)-np.pi
#Ts = dual_fano_transmission(params1, params2, length, λs, intracavity=False, losses=True)
#φs = single_fano_phase(params1, length, λs)-np.pi
#Ts = fano_cavity_transmission(params1, length, λs)
#plt.figure(figsize=(10,7))
#plt.plot(λs, Ts, color="firebrick")
#plt.title("M3 single fano transmission & phase $(l \\approx 10μm)$") 
#plt.xlabel("wavelength [nm]")
#plt.ylabel("normalized transmission [V]", color="firebrick")
#ax2 = plt.gca().twinx()  
#ax2.plot(λs, φs, color="cornflowerblue")
#ax2.set_ylabel("phase [rad]", color='cornflowerblue')
#plt.show()

#### double fano transmission as a function of losses ####

#Ls = [0.0, 0.02, 0.04, 0.06, 0.08] ## loss factor is NOT equal to actual cavity losses
#loss_factor_scan(params1, Ls, λs, lmin=30)


#### double fano transmission as a function of detuning #### 

#Δs = np.linspace(-1.5, 1.5, 5) # low resolution
#Δs = np.linspace(-0.3, 0.3, 5) # high resolution
#Δs = np.linspace(0, 1, 5)
#detuning_plot(Δs, params1, λs, intracavity=False, losses=True, lmin=5)

#### Heat maps of cavity transmission as a function of wavelength and cavity length ####

#l_vs_λ_cmaps(params1, params2, λs, intracavity=False, losses=True, lmin=30)
#double_fano_cmap(params1, params2, λs, intracavity=False, losses=True, lmin=30)


#### Double/single fano cavity transmission plots ####

#length = resonant_cavity_length(params1, λs, lmin=10)
#fano_cavity_transmission_plot(params1, length, λs, intracavity=False, losses=True)

#lmin = 239
#length = (double_cavity_length(params1, params2, λs, lmin=lmin) + double_cavity_length(params2, params1, λs, lmin=lmin))/2
#length = double_cavity_length(params1, params2, λs, lmin=30)
#dual_fano_transmission_plot(params1, params2, length, λs, intracavity=False, losses=True, grating_trans=False, zoom=False)

#Δ = 0.1
#lmin=30
#params2[0] += Δ
#params2[1] += Δ
#ls = np.linspace(double_cavity_length(params1,params2,λs,lmin=lmin), double_cavity_length(params2,params1,λs,lmin=lmin), 5)
#lmins = [10, 40, 100, 200]
#ls = [(double_cavity_length(params1,params2,λs,lmin)+double_cavity_length(params2,params1,λs,lmin))/2 for lmin in lmins]
#cavity_length_plot(ls, params1, params2, λs, intracavity=False, losses=True, zoom=False)


#### for line width comparison of the single and double fano models ####

#grating1 = [951, 951, 0.81, 0.48, 1e-6]
#grating2 = grating1
#lmin = 5
#length = double_cavity_length(params1, params2, λs, lmin=lmin)*0.0 + double_cavity_length(params2, params1, λs, lmin=lmin)*1.0
#print("cavity length: ", length*1e-3)
#line_width_comparison(grating1, grating2, double_cavity_length(grating1, grating2, λs), intracavity=True, losses=False)
#line_width_comparison(params1, params2, double_cavity_length(params1, params2, λs), intracavity=True, losses=True)
#line_width_single(params1, λs)
#line_width_double(params1, params2, λs, length)

#### length scan of the single and double fano cavities

#l=30
#ls = np.linspace(l*1e3, (l+1)*1e3, 10000)
#double_fano_length_scan(params1, params2, ls, λs, plot_both_gratings=True)
#single_fano_length_scan(params1, ls, λs)


#### plotting the calculated reflection/transmission values ####

#theoretical_reflection_values_plot(params3, λs)
#theoretical_reflection_values_comparison_plot(params1, params2, λs)

#peak = fano("/Users/mikkelodeon/optomechanics/Single Fano cavities/Data/M4/70short.txt")
#fitting_params = [950.99,950.99,0.5,1e-2,1e-7]
#params = peak.lossy_fit(fitting_params)

#plt.figure(figsize=(10,6))

#lw_single_fano = line_width_single(M1.lossy_fit([955.5, 955.5, 0.6, 1, 0.1]))

#plt.plot(peak.data[:,0], peak.data[:,1], 'bo', label='transmission data')
#plt.plot(peak.λ_fit, peak.lossy_model(peak.λ_fit, *params), 'cornflowerblue', label='fit: FWHM = %spm' % str(round(2*np.abs(params[3])*1e3,2)))
#plt.plot(λs, Ts, "r.", label="theory (FWHM: %spm)" % str(round(lw_single_fano, 2)))
#plt.xlabel("wavelength [nm]")
#plt.ylabel("normalized ntensity [arb. u.]")
#plt.title("60 μm single fano cavity transmission (M4)")
#plt.legend()
#plt.show()

##### plot linewidth as a function of cavity length #####

#plt.figure(figsize=(10,6))
# 12 -> 21
# ~5um
#cl5 = [5.277452774527745, 5.274415244152443, 5.271347713477135, 5.268325183251831, 5.265172651726517]
#lw5 = [92.6956, 87.0209, 84.2142, 86.8427, 96.6472]
#err5 = [0.2867, 0.1739, 0.1809, 0.2018, 0.178]
# ~30um
#lw30 = [79.8997, 75.7106, 74.117, 76.2582, 79.8249] 
#cl30 = [30.025460254602546, 30.021697716977172, 30.01810018100181, 30.014460144601447, 30.010830108301082]
#err30 = [0.2903, 0.15, 0.0768, 0.0982, 0.3143]
# ~300um
#cl300 = [300.34872348723485, 300.33872338723387, 300.3287232872329, 300.31872318723185, 300.3087230872309] # um 
#lw300 = [31.7342, 29.7608, 28.8908, 29.0817, 30.2533] # pm
#err300 = [0.03, 0.0259, 0.0199, 0.0149, 0.0119]

#plt.errorbar(cavity_lengths, linewidths, errors, fmt="o", color="cornflowerblue", capsize=5)
#plt.title("double Fano linewidth as a function of cavity length: %s" % r"$(l_{M3} + l_{M5})/2$")
#plt.ylabel("FWHM [pm]")
#plt.xlabel("cavity length [μm]")      
#plt.show()

#linewidth_length_plot(params1, params2, λs)

#theoretical_phase_plot(params1, λs)
#length = resonant_cavity_length(params1, λs, lmin=30)
#single_fano_phase_plot(params1, length, λs)

#lmin=30
#length = (double_cavity_length(params1, params2, λs, lmin=lmin) + double_cavity_length(params2, params1, λs, lmin=lmin))/2
#length = double_cavity_length(params2, params1, λs, lmin=lmin)
#double_fano_phase_plot(params1, params2, length, λs)

#lmin=20
#length_M3 = (double_cavity_length(params1, params2, λs, lmin=lmin))
#Ts_M3 = dual_fano_transmission(params1, params2, length_M3, λs, loss_factor=0.05)

#length_M5 = double_cavity_length(params2, params1, λs, lmin=lmin)
#Ts_M5 = dual_fano_transmission(params1, params2, length_M5, λs, loss_factor=0.05)

#length_mid = (double_cavity_length(params1, params2, λs, lmin=lmin)*0.5 + double_cavity_length(params2, params1, λs, lmin=lmin)*0.5)
#Ts_mid = dual_fano_transmission(params1, params2, length_mid, λs, loss_factor=0.05)

lmin = 67
length = (double_cavity_length(params1, params2, λs, lmin=lmin)*0.8 + double_cavity_length(params2, params1, λs, lmin=lmin)*0.2)
print(length)
Ts = dual_fano_transmission(params1, params2, length, λs, loss_factor=0.05, intracavity=False)

p0 = [0, 0, 0, 951.7, 100e-3]
#bounds = [[0,0,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
#p0 = [951.7,951.7,0.6,0.1,1e-6]

popt, pcov = curve_fit(fit_model, λs, Ts, p0=p0, maxfev=100000)
err = np.sqrt(np.diag(pcov))
lw_err = round(err[4]*1e3,3)
lw = round(popt[4]*1e3,3)
legend = [lw, lw_err, round(length*1e-3,3)]
print("lw error: ", lw_err)
print(popt)

xs = np.linspace(λs[0], λs[-1], 1000)
 
plt.figure(figsize=(10,6))
plt.scatter(λs, Ts, color="royalblue", label="theory")
plt.plot(xs, fit_model(xs, *popt), color="cornflowerblue", label="fit: HWHM $\\approx$ %5.3f +/- %5.3fpm, cavity length $\\approx$ %5.3fμm" % tuple(legend))
#plt.plot(λs, Ts_M3, color="tomato", linestyle="-.", label="theory, $l = l_{M3}$")
#plt.plot(λs, Ts_M5, color="seagreen", linestyle="-.", label="theory, $l = l_{M5}$")
#plt.plot(λs, Ts_mid, color="royalblue", linestyle="--", label="theory, $l = (l_{M3} + l_{M5})/2$")
#plt.scatter(data[:,0], data[:,1], marker='.', color="maroon", label="data", zorder=4)
plt.title("M3/M5 double fano transmission") 
plt.xlabel("wavelength [nm]")
plt.ylabel("normalized transmission [V]")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
plt.subplots_adjust(bottom=0.2)
plt.show()

