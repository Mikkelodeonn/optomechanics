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

G2 = M3.lossy_fit([952,952,0.6,1,0.1])
G1 = M5.lossy_fit([952,952,0.6,1,0.1])
#Δ = 0.0
#params1 = [951.206, 951.356, 0.807, 0.528, 1e-10]
#params2 = [951.206+Δ, 951.356+Δ, 0.807, 0.528, 1e-10]
#print("params1: ", params1)
#print("params2: " ,params2)

## grating parameters -> [λ0, λ1, td, γλ, α]
# λ0 -> resonance wavelengths
# λ1 -> guided mode resonance wavelength
# td -> direct transmission coefficient
# γλ -> width of guided mode resonance
# α  -> loss factor 

#data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/20um/20s5.txt")
#PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/20um/20s5_PI.txt")
#norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/normalization/short_scan.txt")
#norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250225/normalization/short_scan_PI.txt")
#PI_0 = PI_data[0,1]
#data[:,1] = [dat/(PI/PI_0) for dat,PI in zip(data[:,1], PI_data[:,1])]
#norm[:,1] = [N/(PI/PI_0) for N,PI in zip(norm[:,1], norm_PI[:,1])]

#data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])]

mist_data = np.loadtxt("/Users/mikkelodeon/optomechanics/MIST/data/MIST_ref.txt", skiprows=1)
mist_ref = [np.zeros(len(mist_data)),np.zeros(len(mist_data))]
mist_ref[0] = [d[0]*1e3 for d in mist_data]; mist_ref[1] = [d[1] for d in mist_data]

mist_trans = np.copy(mist_ref)
mist_trans[1] = [1-r for r in mist_trans[1]]
eps = 0.01
mist_t = [(1-eps)*t+eps for t in mist_trans[1]]

#λs = np.array(mist_ref[0])

#λs = np.linspace(, 952.5, 500)
#λs = np.linspace(951.65, 951.95, 500)
λs = np.linspace(M3.data[:,0][0], M3.data[:,0][-1], 10000)
#λs = np.linspace(data[:,0][0], data[:,0][-1], 1000)
#λs = np.linspace(950.5, 953, 10000)
lmin = 29.9
L = 0.016
Δ = 0.3
#λs = np.linspace(950.500, 951.500, 90) # 950, 952
#λs = np.linspace(910, 980, 10000)
#λs = np.linspace(951.68, 951.90, 200)

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def lossless_model(λ, λ0, λ1, td, γλ): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def fit_model(λ, a, b, c, λ0, δλ): ## generel fano model
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

def theoretical_phase_values(params: list, losses=True, loss_factor=0.05):
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

    print(popt_t1, popt_t2)

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

def broadband_cavity_length(λ0: float, R: float, lmin=50):
    lengths = []
    Ts = []

    #tm = np.sqrt(1-R)
    #rm = np.sqrt(R)
    tm = 0.809383525
    rm = np.sqrt(1 - tm**2)

    ls = list(np.linspace(lmin,lmin+1,100000)*1e3)

    for l in ls:
        λ = λ0
        t = np.abs(tm**2*np.exp(1j*(2*np.pi/λ)*l)/(1-rm**2*np.exp(2j*(2*np.pi/λ)*l)))**2
        Ts.append(t)

    peak_indices = find_peaks(Ts)

    for idx in peak_indices[0]:
        lengths.append(ls[idx])

    if np.abs(Ts[ls.index(lengths[0])] - Ts[ls.index(lengths[1])]) > 1e10:
        resonance_length = lengths[1]
    else:
        resonance_length = lengths[0]

    return resonance_length 

def broadband_transmission(λ, l, R=0.99): ## lossless
    #tm = np.sqrt(1-R)
    #rm = np.sqrt(R)
    tm = 0.809383525
    rm = np.sqrt(1 - tm**2)
    T = np.abs(tm**2*np.exp(1j*(2*np.pi/λ)*l)/(1-rm**2*np.exp(2j*(2*np.pi/λ)*l)))**2
    return T 

def resonant_cavity_length(params: list, λs: np.array, lmin=50, losses=True):
    reflection_values = theoretical_reflection_values(params, losses=losses)[1]
    transmission_values = np.sqrt(model(λs, *params))

    reflection_values = [complex(r) for r in reflection_values]
    transmission_values = [complex(t) for t in transmission_values]

    idx = np.argmin(transmission_values)

    lengths = []
    Ts = []

    tg = transmission_values[idx]
    rg = reflection_values[idx]
    tm = np.sqrt(0.01)
    rm = np.sqrt(0.99)

    ls = list(np.linspace(lmin,lmin+1,10000)*1e3)

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

def double_cavity_length(params1: list, params2: list, λs: np.array, lmin=50, loss_factor=0.05, losses=True):
    r1 = theoretical_reflection_values(params1, losses=losses, loss_factor=loss_factor)[1]
    r2 = theoretical_reflection_values(params2, losses=losses, loss_factor=loss_factor)[1]
    t1 = np.sqrt(model(λs, *params1))
    t2 = np.sqrt(model(λs, *params2))

    r1 = [complex(r) for r in r1]; r2 = [complex(r) for r in r2]
    t1 = [complex(t) for t in t1]; t2 = [complex(t) for t in t2]

    idx = np.argmin(np.array(t1))

    rg1 = r1[idx]; rg2 = r2[idx]
    tg1 = t1[idx]; tg2 = t2[idx]

    lengths = []
    Ts = []

    ls = list(np.linspace(lmin,lmin+1,10000)*1e3)

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

    plt.figure(figsize=(10,7))
    if plot_both_gratings == True:
        plt.plot(ls*1e-3, Ts, "royalblue", label="$l_{res,M3}$: %sμm" % str(round(resonance_length,3)))
        plt.plot(ls*1e-3, Ts_2, "--", color="lightcoral", label="$l_{res,M5}$: %sμm" % str(round(resonance_length_2,3)))
    else:
        plt.plot(ls*1e-3, Ts, "royalblue", label="$l_{res}$: %sμm" % str(round(resonance_length,3)))
    #plt.title("Double fano cavity transmission as a function of cavity length")
    plt.xlabel("cavity length [μm]", fontsize=24)
    plt.ylabel("transmission [arb. u.]", fontsize=24)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
    plt.subplots_adjust(bottom=0.3)
    plt.show()
 
def fano_cavity_transmission(params: list, length: np.array, λs: np.array, intracavity=False, losses=True):
    #print("single fano length:", length)

    reflection_values = theoretical_reflection_values(params, losses=losses)[1]
    transmission_values = np.sqrt(model(λs, *params))

    if intracavity == False:
        def cavity_transmission(λ, rg, tg, l):
            tm = np.sqrt(0.01)
            rm = np.sqrt(0.99)
            T = np.abs(tm*tg*np.exp(1j*(2*np.pi/λ)*l)/(1-rm*rg*np.exp(2j*(2*np.pi/λ)*l)))**2
            return T 
        
    if intracavity == True:
        def cavity_transmission(λ, rg, tg, l):
            rm = np.sqrt(0.95)
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
        rg1 = theoretical_reflection_values(params1, losses=losses)[0]
        rg2 = theoretical_reflection_values(params2, losses=losses)[0]
        ax.plot(λs, tg1, "maroon", linestyle="--", alpha=0.7, label="$|t_{g}|^2$")
        ax.plot(λs, tg2, "indianred", linestyle="--", alpha=0.7, label="$|t_{g}^{\\prime}|^2$")
        ax.plot(λs, rg1, "steelblue", linestyle="--", alpha=0.7, label="$|r_{g}|^2$")
        ax.plot(λs, rg2, "lightskyblue", linestyle="--", alpha=0.7, label="$|r_{g}^{\\prime}|^2$")
        x1, x2, y1, y2 = params1[0]-1.5, params1[0]+1.5, 0.01, 1.03
        axins = ax.inset_axes([0.02, 0.05, 0.45, 0.45])
        axins.plot(λs, Ts, color="forestgreen")
        axins.plot(λs, tg1, "maroon", linestyle="--", alpha=0.7)
        axins.plot(λs, tg2, "indianred", linestyle="--", alpha=0.7)
        axins.plot(λs, rg1, "steelblue", linestyle="--", alpha=0.7)
        axins.plot(λs, rg2, "lightskyblue", linestyle="--", alpha=0.7)
        axins.set_xlim(x1,x2)
        axins.set_ylim(y1,y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        mark_inset(ax, axins, loc1=2, loc2=4, edgecolor="black", alpha=0.3)
    #ax.set_title("Double fano transmission as a function of wavelength ($\\left(l_{M3} + l_{M5}\\right)/2 \\approx$%sμm)" % str(round(length*1e-3,3)))
    #ax.set_title("Double fano transmission as a function of wavelength ($l_{M3} \\approx$%sμm)" % str(round(length*1e-3,3)))
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("normalized transmission [arb.u.]")
    ax.plot(λs, Ts, "forestgreen", label="$|E_{out}|^2 / |E_{0,in}|^2$")
    ax.legend(loc = "upper right")
    plt.show()

def detuning_plot(Δs: list, params: list, λs: np.array, intracavity=False, losses=True, lmin=30): ## plots dual fano cavity transmission for different values for the detuning
    plt.figure(figsize=(10,7))
    #length = double_cavity_length(params, params, λs, lmin=lmin)
    linestyles = ["-", "--", "-.", ":", ":"]
    colors = ["forestgreen","royalblue","darkgoldenrod", "darkorange", "firebrick"]
    tmax = []
    for Δ, paint, style in zip(Δs,colors,linestyles):
        params2 = np.copy(params)
        params2[0] += Δ
        params2[1] += Δ
        length = (double_cavity_length(params, params2, λs, lmin=lmin, losses=losses) + double_cavity_length(params2, params, λs, lmin=lmin, losses=losses))/2
        Ts =  dual_fano_transmission(params, params2, length, λs, intracavity=intracavity, losses=losses)
        #if paint == "forestgreen":
        #    tmax.append(np.max(Ts))
        #if np.abs(Δ) < 1e-6:
        #    linesize = 3
        #else:
        #    linesize = 2
        #plt.scatter(Δ, np.max(Ts), color=paint, label="Δ=%snm" %(round(Δ,2)))
        plt.plot(λs, Ts/np.max(Ts), color=paint, linestyle=style, label="Δ=%snm" %(round(Δ,2)))
    #plt.title("Double fano transmission for varying detuning Δ (cavity length %s)" %(r"$\rightarrow l_{g,1} \approx 100 \mu m$"))
    #plt.title("Double fano transmission for varying detuning Δ (cavity length %s)" %(r"$\rightarrow (l_{g,1} + l_{g,2})/2 \approx 10 \mu m$"))
    plt.xlabel("wavelength [nm]", fontsize=28)
    #plt.xlabel("Δ [nm]", fontsize=28)
    #plt.ticklabel_format(style='sci', axis="y", scilimits=(0,0))
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    #plt.ylabel("max intensity [arb. u.]", fontsize=28)
    plt.ylabel("norm. intensity [arb. u.]", fontsize=28)
    plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(bottom=0.3)
    plt.locator_params(axis='x', tight=True, nbins=8)
    plt.show()

def loss_factor_scan(params: list, loss_list: list, λs: np.array, lmin=50): ## only makes sense for symmetric double fano
    plt.figure(figsize=(10,7))
    linestyles = ["-.", "--", "-", "--", "-."]
    colors = ["skyblue","lightcoral","lightgreen", "firebrick", "forestgreen"]
    for loss, linestyle, color in zip(loss_list, linestyles, colors):
        r_max = np.max(theoretical_reflection_values(params, loss_factor=loss)[0])
        t_min = np.min(np.real(model(λs, *params)))
        percentile_loss = round((1-r_max-t_min)*1e2,2) ## in percent
        length = double_cavity_length(params, params, λs, lmin=lmin, loss_factor=loss)
        Ts = dual_fano_transmission(params1, params2, length, λs, loss_factor=loss)
        plt.plot(λs, Ts, color=color, linestyle=linestyle, linewidth=2, label="L = %s%%" %str(2*np.abs(percentile_loss)))
    #plt.title("symmetric double fano transmission for different cavity losses, length: $l_{M3} \\approx$ %s $\mu m$" % str(lmin))
    plt.xlabel("wavelength [nm]", fontsize=28)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    #plt.ylabel("norm. ref./trans. [arb. u.]", fontsize=28)
    plt.ylabel("norm. trans. [arb. u.]", fontsize=28)
    plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(bottom=0.3)
    plt.locator_params(axis='x', tight=True, nbins=8)
    plt.show()

def cavity_length_plot(ls: list, params1: list, params2: list, λs: np.array, intracavity=False, losses=True, zoom=False):
    fig, ax = plt.subplots(figsize=(10,7))
    linestyles = ["-.", "--", "-", "--", "-."]
    #linestyles = [":", "-.", "--", "-"]
    colors = ["skyblue","royalblue","forestgreen", "firebrick", "lightcoral"]
    #colors = ["skyblue", "lightgreen", "lightcoral", "royalblue"]
    Ts_inset = []
    for l, paint, style in zip(ls, colors, linestyles):
        Ts = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
        ax.plot(λs, Ts, color=paint, linestyle=style, label="l = %sμm" % str(round(l*1e-3,3)))
        Ts_inset.append(Ts)
    if zoom == True:
        axins = ax.inset_axes([0.05, 0.05, 0.35, 0.35])
        for Ts, paint, style in zip(Ts_inset,colors,linestyles):
            axins.plot(λs, Ts, color=paint, linestyle=style)
        axins.set_xlim(950.50, 951.75)
        #axins.set_xlim(950, 953.5)
        axins.set_ylim(0.02, 1.04)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        mark_inset(ax, axins, loc1=2, loc2=4, edgecolor="black", alpha=0.3)
    #plt.title("Double fano transmission for different cavity lengths %s" %("$l_{M3} \\rightarrow l_{M5}$")) 
    #plt.title("Double fano transmission for different cavity lengths:  %s" %("$(l_{M3} + l_{M5})/2$")) 
    plt.xlabel("wavelength [nm]", fontsize=28)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.ylabel("norm. trans. [arb. u.]", fontsize=28)
    plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(bottom=0.3)
    plt.locator_params(axis='x', tight=True, nbins=8)
    plt.show()

def l_vs_λ_cmaps(params1: list, params2: list, λs: np.array, intracavity=False, losses=True, lmin=30): 
    params2[1] += 0
    params2[0] += 0
    Δs = 0.15
    rows = 3
    columns = 3
    Δ_label = -0.14
    fig, ax = plt.subplots(rows,columns, figsize=(18,8))
    for i in range(rows):
        for j in range(columns):
            params2[1] += Δs
            params2[0] += Δs
            Δ_label += Δs
            if np.abs(Δ_label) < 1e-6:
                ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin, losses=losses)-0.1, double_cavity_length(params2, params1, λs, lmin=lmin, losses=losses)+0.1, 20)
            else:
                ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin, losses=losses), double_cavity_length(params2, params1, λs, lmin=lmin, losses=losses), 100)
            Ts = []
            for l in ls:
                T = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
                Ts.append(T)

            cmap = np.zeros([len(Ts),len(Ts[0])])

            for h in range(len(Ts)):
                for k in range(len(Ts[h])):
                    cmap[h,k] = Ts[h][k] 
            
            l_labels = [round(l*1e-3,4) for l in np.linspace(np.min(ls),np.max(ls),10)]
            λ_labels = np.linspace(np.min(λs), np.max(λs),10)
            λ_labels = [round(label,2) for label in λ_labels]

            im = ax[i,j].imshow(cmap, aspect="auto", extent=[np.min(λs), np.max(λs), np.min(ls), np.max(ls)])
            ax[i,j].set_title("Δ = %snm" %(round(Δ_label,2)), fontsize=16)
            #ax[i,j].set_xticks(np.linspace(np.min(λs), np.max(λs),10))
            #ax[i,j].set_xticklabels(λ_labels, fontsize=5)
            #ax[i,j].set_yticks(np.linspace(np.min(ls), np.max(ls),10))
            #ax[i,j].set_yticklabels(l_labels, fontsize=5)    
            ax[i,j].set_xticks(np.linspace(np.min(λs), np.max(λs),10))
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticks(np.linspace(np.min(ls), np.max(ls),10))
            ax[i,j].set_yticklabels([]) 
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    #if intracavity == False and losses == False:
    #    fig.text(0.5, 0.93, 'Double fano lossless transmission as a function of cavity length for different values of Δ', ha='center', va='center', fontsize=16) 
    #elif intracavity == False and losses == True:
    #    fig.text(0.5, 0.93, 'Double fano transmission as a function of cavity length for different values of Δ', ha='center', va='center', fontsize=16) 
    #else: 
    #    fig.text(0.5, 0.93, 'Double fano lossless intracavity intensity as a function of cavity length for different values of Δ', ha='center', va='center', fontsize=16) 
    fig.text(0.5, 0.06, 'wavelength [nm]', ha='center', va='center', fontsize=28)
    fig.text(0.08, 0.5, 'cavity length [μm]', ha='center', va='center', fontsize=28, rotation="vertical")
    plt.show()

def double_fano_cmap(params1: list, params2: list, λs: np.array, intracavity=False, losses=True, lmin=50):

    plt.figure(figsize=(10,6))

    ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin, losses=losses), double_cavity_length(params2, params1, λs, lmin=lmin, losses=losses),20)
    Ts = []
    for l in ls:
        T = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
        Ts.append(T)

    cmap = np.zeros([len(Ts),len(Ts[0])])

    for h in range(len(Ts)):
        for k in range(len(Ts[h])):
            cmap[h,k] = Ts[h][k] 

    l_labels = [round(l*1e-3,4) for l in ls]
    λ_labels = np.linspace(np.min(λs), np.max(λs),10)
    λ_labels = [round(label,2) for label in λ_labels]

    length1 = 0.5*double_cavity_length(params1, params2, λs, lmin=lmin, losses=False) + 0.5*double_cavity_length(params2, params1, λs, lmin=lmin, losses=False)
    length2 = 0.2*double_cavity_length(params1, params2, λs, lmin=lmin, losses=False) + 0.8*double_cavity_length(params2, params1, λs, lmin=lmin, losses=False)
    length3 = 0.8*double_cavity_length(params1, params2, λs, lmin=lmin, losses=False) + 0.2*double_cavity_length(params2, params1, λs, lmin=lmin, losses=False)

    l1 = [length1 for _ in np.zeros(len(λs))]
    l2 = [length2 for _ in np.zeros(len(λs))]
    l3 = [length3 for _ in np.zeros(len(λs))]

    plt.plot(λs, l1, color="lime", linestyle="-", lw=4)
    plt.plot(λs, l2, color="magenta", linestyle="--", lw=4)
    plt.plot(λs, l3, color="cyan", linestyle="-.", lw=4)
    plt.imshow(cmap, aspect="auto", extent=[np.min(λs), np.max(λs), np.min(ls), np.max(ls)])
    plt.xticks(λ_labels, fontsize=21)
    plt.yticks(ls, l_labels, fontsize=21)
    ax = plt.gca()
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(axis='x', direction='out', which='both')
    plt.xlabel("wavelength [nm]", fontsize=28)
    plt.ylabel("cavity length [μm]", fontsize=28) 
    #plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.locator_params(axis='both', nbins=8)
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
    lengths = np.linspace(double_cavity_length(params1, params2, λs, lmin=29.9, losses=losses), double_cavity_length(params2, params1, λs, lmin=29.9, losses=losses), 11)

    for length in lengths:
        Ts =  dual_fano_transmission(params1, params2, length, λs, intracavity=intracavity, losses=losses)
        popt, pcov = curve_fit(model, λs, Ts, p0=(np.array(params1)+np.array(params2))/2, maxfev=10000)
        HWHM = np.abs(popt[3])*1e3
        err = np.sqrt(np.diag(pcov))[3]*1e3
        lws.append(HWHM)
        errs.append(err)
    
    plt.figure(figsize=(10,7))
    plt.errorbar(lengths*1e-3, lws, errs, fmt="o", capsize=3, color="royalblue")
    #plt.title("intracavity HWHM as a function of cavity length ($l = l_{M3} \\rightarrow l_{M5} \\approx 20μm$)")
    plt.xlabel("cavity length [μm]", fontsize=28)
    plt.ylabel("HWHM [pm]", fontsize=28)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.subplots_adjust(bottom=0.2)
    plt.locator_params(axis='x', tight=True, nbins=8)
    plt.show()

params1, _ = curve_fit(model, mist_trans[0], mist_t, p0=[951.2, 951.2, 0.7, 0.5, 1e-6], maxfev=10000)
#r_params, _ = curve_fit(model, mist_trans[0], mist_ref[1], p0=[951.2, 951.2, 0.7, 0.5, 1e-6], maxfev=10000)
#print("rd=",r_params[3])
#print("params1:", params1)
#params1 = [951.208053, 951.355926, 0.809383525, 0.527290186, 4.42286435e-07]
params2 = np.copy(params1)
params2[0] += Δ
params2[1] += Δ

λs_fit = np.linspace(λs[0], λs[-1], 10000)

broadband_length = broadband_cavity_length(params1[0], R=0.99, lmin=lmin)
print("broadband length: ", broadband_length*1e-3, "μm")
single_length = resonant_cavity_length(params1, λs, lmin=lmin, losses=False)
print("single fano length: ", single_length*1e-3, "μm") 
double_length = 0.5*double_cavity_length(params1, params1, λs, lmin=lmin, losses=False) + 0.5*double_cavity_length(params1, params1, λs, lmin=lmin, losses=False)
print("double fano length", double_length*1e-3, "μm")

bb_ts = broadband_transmission(λs, broadband_length)
#bb_ts2 = broadband_transmission(λs, single_length, 0.90)
single_ts = fano_cavity_transmission(params1, single_length, λs, intracavity=False, losses=False)
double_ts = dual_fano_transmission(params1, params1, double_length, λs, intracavity=False, losses=False)

#Ls = [0.0, 0.0025, 0.005, 0.0075, 0.01] ## loss factor is NOT equal to actual cavity losses
#loss_factor_scan(params1, Ls, λs, lmin=30)

fig, ax = plt.subplots(figsize=(10,7))


#### double fano transmission as a function of detuning #### 

#Δs = np.linspace(-1.5, 1.5, 5) # low resolution
#Δs = np.linspace(-0.3, 0.3, 5) # high resolution
#Δs = np.linspace(0, 1, 5)
#detuning_plot(Δs, params1, λs, intracavity=True, losses=False, lmin=29)

#### Heat maps of cavity transmission as a function of wavelength and cavity length ####

#l_vs_λ_cmaps(params1, params2, λs, intracavity=False, losses=False, lmin=lmin)
#double_fano_cmap(params1, params2, λs, intracavity=False, losses=False, lmin=lmin)

#### length scan of the single and double fano cavities

#l=29.9
#ls = np.linspace(l*1e3, (l+1)*1e3, 10000)
#double_fano_length_scan(params1, params2, ls, λs, plot_both_gratings=False)
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

#linewidth_length_plot(params1, params2, λs, intracavity=True, losses=False)

#theoretical_phase_plot(params1, λs)
#length = resonant_cavity_length(params1, λs, lmin=30)
#single_fano_phase_plot(params1, length, λs)

#length = (double_cavity_length(params1, params2, λs, lmin=lmin) + double_cavity_length(params2, params1, λs, lmin=lmin))/2
#length = double_cavity_length(params2, params1, λs, lmin=lmin)
#double_fano_phase_plot(params1, params2, length, λs)

#M3 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M3_trans.txt")
#M3_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M3_trans_PI.txt")
#M3_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/grating_trans.txt")
#M3_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/grating_trans_PI.txt")

#M5 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M5_trans.txt")
#M5_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M5_trans_PI.txt")
#M5_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/grating_trans.txt")
#M5_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/grating_trans_PI.txt")

#M3[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M3[:,1], M3_PI[:,1], M3_norm[:,1], M3_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
#M5[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5[:,1], M5_PI[:,1], M5_norm[:,1], M5_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

#λs = np.linspace(M3[:,0][0], M3[:,0][-1], 50)
#λs_fit = np.linspace(M3[:,0][0], M3[:,0][-1], 10000)

#p0 = [952,952,0.6,1,0.1]
#params1, pcov1 = curve_fit(model, M3[:,0], M3[:,1], p0=p0)
#params2, pcov2 = curve_fit(model, M5[:,0], M5[:,1], p0=p0)

#lmin=128
#length_M3 = (double_cavity_length(params1, params2, λs, lmin=lmin))
#Ts_M3 = dual_fano_transmission(params1, params2, length_M3, λs, loss_factor=0.05)

#length_M5 = double_cavity_length(params2, params1, λs, lmin=lmin)
#Ts_M5 = dual_fano_transmission(params1, params2, length_M5, λs, loss_factor=0.05)

#length_mid = (double_cavity_length(params1, params2, λs, lmin=lmin)*0.5 + double_cavity_length(params2, params1, λs, lmin=lmin)*0.5)
#Ts_mid = dual_fano_transmission(params1, params2, length_mid, λs, loss_factor=0.05)

#lmin = 718.7
#length = (double_cavity_length(params1, params2, λs, lmin=lmin)*0.9 + double_cavity_length(params2, params1, λs, lmin=lmin)*0.1)
#print(length)
#Ts = dual_fano_transmission(params1, params2, length, λs, loss_factor=0.05, intracavity=False)

#p0 = [0, 0, 0, 951.7, 100e-3]
#bounds = [[0,0,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
#p0 = [951.7,951.7,0.6,0.1,1e-6]

#popt, pcov = curve_fit(fit_model, λs, Ts, p0=p0, maxfev=100000)
#err = np.sqrt(np.diag(pcov))
#lw_err = round(err[4]*1e3,3)
#lw = np.abs(round(popt[4]*1e3,3))
#legend = [lw, lw_err, round(length*1e-3,3)]
#print("lw error: ", lw_err)
#print(popt)

#xs = np.linspace(λs[0], λs[-1], 1000)
 
#plt.figure(figsize=(10,6))
#plt.scatter(λs, Ts, color="royalblue", label="theory")
#plt.plot(xs, fit_model(xs, *popt), color="cornflowerblue", label="fit: HWHM $\\approx$ %5.3f +/- %5.3fpm, cavity length $\\approx$ %5.3fμm" % tuple(legend))
#plt.plot(λs, Ts_M3, color="tomato", linestyle="-.", label="theory, $l = l_{M3}$")
#plt.plot(λs, Ts_M5, color="seagreen", linestyle="-.", label="theory, $l = l_{M5}$")
#plt.plot(λs, Ts_mid, color="royalblue", linestyle="--", label="theory, $l = (l_{M3} + l_{M5})/2$")
#plt.scatter(data[:,0], data[:,1], marker='.', color="maroon", label="data", zorder=4)
#plt.title("M3/M5 double fano transmission $(l = 9/10 \\cdot l_{M3} + 1/10 \\cdot l_{M5})$") 
#plt.scatter(M3[:,0], M3[:,1], label="M3 (top)")
#plt.scatter(M5[:,0], M5[:,1], label="M5 (bottom)")
#plt.plot(λs_fit, model(λs_fit, *params1), label="lw: %s(M3)" % (params1[0]))
#plt.plot(λs_fit, model(λs_fit, *params2), label="lw: %s(M5)" % (params2[0]))
#plt.xlabel("wavelength [nm]")
#plt.ylabel("normalized transmission [V]")
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
#plt.subplots_adjust(bottom=0.2)
#plt.show()

## examined lengths for simulated linewidths comparison: 10, 30, 90, 270, 810


double_length1 = 0.5*double_cavity_length(params1, params2, λs, lmin=lmin, losses=False) + 0.5*double_cavity_length(params2, params1, λs, lmin=lmin, losses=False)
double_length2 = 0.2*double_cavity_length(params1, params2, λs, lmin=lmin, losses=False) + 0.8*double_cavity_length(params2, params1, λs, lmin=lmin, losses=False)
double_length3 = 0.8*double_cavity_length(params1, params2, λs, lmin=lmin, losses=False) + 0.2*double_cavity_length(params2, params1, λs, lmin=lmin, losses=False)

double_ts1 = dual_fano_transmission(params1, params2, double_length1, λs, intracavity=False, losses=False)
double_ts2 = dual_fano_transmission(params1, params2, double_length2, λs, intracavity=False, losses=False)
double_ts3 = dual_fano_transmission(params1, params2, double_length3, λs, intracavity=False, losses=False)

#plt.plot(λs, double_ts1, linestyle="-", lw=3, color="lime")
#plt.plot(λs, double_ts2, linestyle="--", lw=3, color="magenta")
#plt.plot(λs, double_ts2, linestyle="-.", lw=3, color="cyan")

rs1 = [float(r) for r in theoretical_reflection_values(params1, losses=False, loss_factor=0.03)[0]]
grating_trans1 = model(λs, *params1)
grating_losses = [1-r-t for r,t in zip(rs1, grating_trans1)]
rs2 = [float(r) for r in theoretical_reflection_values(params2, losses=False)[0]]
grating_trans2 = model(λs, *params2)


ts = dual_fano_transmission(params1, params1, double_length, λs, intracavity=False, losses=True, loss_factor=L)
r_max = np.max(theoretical_reflection_values(params1, loss_factor=L)[0])
t_min = np.min(np.real(model(λs, *params1)))
percentile_loss = round((1-r_max-t_min)*1e2,2) ## in percent

losses = [0.0, 0.02, 0.06, 0.12, 0.24, 0.48] ## in percent
lws = [26.9, 32.126, 42.732, 59.052, 93.306, 169.794] ## in pm

popt4, _4 = curve_fit(model, λs, ts, p0=params1, maxfev=10000)


#plt.scatter(λs, ts, color="forestgreen", marker=".", label="double fano: $L_{total}$ = %s%%" %str(2*abs(percentile_loss)))
#plt.plot(λs_fit, model(λs_fit, *popt4), color="forestgreen", alpha=0.5, label="$HWHM_{double} \\approx$ %spm" % str(round(abs(popt4[3])*1e3,3)))

## Figure convention -> double fano: forestgreen, single fano: orangered, broadband: royalblue, grating_ref: blueviolet

popt1, pcov = curve_fit(model, λs, double_ts1, p0=[951,951,0.6,0.1,1e-6])
popt2, pcov = curve_fit(model, λs, double_ts2, p0=[951,951,0.6,0.1,1e-6])
popt3, pcov = curve_fit(model, λs, double_ts3, p0=[951,951,0.6,0.1,1e-6])
lw1 = round(abs(popt1[3])*1e3,1)
lw2 = round(abs(popt2[3])*1e3,1)
lw3 = round(abs(popt3[3])*1e3,1)
print("HWHM (lime) =     ", lw1, "μm")
print("HWHM (magenta) =  ", lw2, "μm")
print("HWHM (cyan) =     ", lw3, "μm")

#plt.scatter(λs, double_ts1, marker=".", color="lime")
#plt.scatter(λs, double_ts2, marker=".", color="magenta")
#plt.scatter(λs, double_ts3, marker=".", color="cyan")
#plt.plot(λs_fit, model(λs_fit, *popt1), color="lime", linestyle="-", alpha=0.5, label="$HWHM \\approx$ %spm" % str(lw1))
#plt.plot(λs_fit, model(λs_fit, *popt2), color="magenta", linestyle="--", alpha=0.5, label="$HWHM\\approx$ %spm" % str(lw2))
#plt.plot(λs_fit, model(λs_fit, *popt3), color="cyan",linestyle="-.", alpha=0.5, label="$HWHM \\approx$ %spm" % str(lw3))
#plt.plot(λs, double_ts1, color="lime", label="$l = 0.5 l + 0.5l^{\\prime} \\approx %s \\mu m$" % str(round(double_length1*1e-3,3)))
#plt.plot(λs, double_ts2, color="magenta", linestyle="--", label="$l = 0.2 l + 0.8l^{\\prime} \\approx %s \\mu m$" % str(round(double_length2*1e-3,3)))
#plt.plot(λs, double_ts3, color="cyan", linestyle="-.", label="$l = 0.8 l + 0.2l^{\\prime} \\approx %s \\mu m$" % str(round(double_length3*1e-3,3)))

#popt, pcov = curve_fit(model, λs, bb_ts, p0=params1, maxfev=10000)
#popt1, _ = curve_fit(model, λs, single_ts, p0=params1, maxfev=10000)
#popt2, _1 = curve_fit(model, λs, double_ts, p0=params1, maxfev=10000)
#popt3, _2 = curve_fit(model, λs, rs2, p0=params1, maxfev=10000)

#grating_losses_plot, λs_losses = zip(*[(l, λ) for l, λ in zip(grating_losses, λs) if l > 1e-2])

#print(min(grating_trans1) + max(rs1) + max(grating_losses))

#plt.plot(λs, bb_ts, color="royalblue", linestyle="--", label="broadband trans.")
#plt.plot(λs, single_ts, color="orangered", linestyle="--", label="single Fano trans.", zorder=3)
#plt.plot(λs, double_ts, color="forestgreen", label="$|E_{out}|^2/|E_{0,in}|^2$")
#plt.plot(λs, rs1, color="steelblue", linestyle="--", label="$|r_g|^2$")
#plt.plot(λs, rs2, color="lightskyblue", linestyle="--", label="$|r_g^{\prime}|^2$")
#x1, x2, y1, y2 = params1[0]-0.15, params1[0]+0.15, 0.01, 1.03
#axins = ax.inset_axes([0.02, 0.70, 0.4, 0.4])
#axins.plot(λs[5250:5750], double_ts[5250:5750], color="forestgreen")
#axins.plot(λs[5250:5750], single_ts[5250:5750], "orangered", linestyle="--")
#axins.plot(λs[5250:5750], rs1[5250:5750], "blueviolet", linestyle="-.")
#axins.set_xticklabels([])
#axins.set_yticklabels([])
#mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="black", alpha=0.3)
#plt.plot(λs, grating_trans1, color="maroon", label="transmission: $|t_{min}|^2 =$ %s%%" %str(round(min(grating_trans1)*1e2,2)))
#plt.scatter(λs, rs1, color="steelblue", marker=".")#, label="reflectivity: $|r_{max}|^2 =$ %s%%" % str(round(max(rs1)*1e2,2)))
#plt.scatter(λs, grating_trans1, color="maroon", marker=".")#, label="transmission: $|t_{min}|^2 =$ %s%%" %str(round(min(grating_trans1)*1e2,2)))
#plt.plot(λs, grating_losses, color="blueviolet", label="losses: $L_{max} =$ %s%%" % str(round(max(grating_losses)*1e2,2)))
#plt.scatter(λs, rs2, color="lightskyblue", marker=".")
#plt.scatter(λs, grating_trans2, color="indianred", marker=".")

#popt, _ = curve_fit(model, λs, grating_trans1, p0=[951,951,0.6,0.1,1e-6], maxfev=10000)
#popt1, _ = curve_fit(model, λs, grating_trans2, p0=[951,951,0.6,0.1,1e-6], maxfev=10000)
#popt2, _ = curve_fit(model, λs, rs1, p0=[951,951,0.6,0.1,1e-6], maxfev=10000)
#popt3, _ = curve_fit(model, λs, rs2, p0=[951,951,0.6,0.1,1e-6], maxfev=10000)

#plt.plot(λs_fit, model(λs_fit, *popt2), alpha=0.6, color="steelblue")#, label="ref.: $\\lambda_{0} = %snm$" % str(round(popt2[0],2)))
#plt.plot(λs_fit, model(λs_fit, *popt), alpha=0.6, color="maroon", label="$\\lambda_{0} = %snm$" % str(round(popt[0],2)))
#plt.plot(λs_fit, model(λs_fit, *popt3), alpha=0.6, color="lightskyblue")#, label="detuned ref.: $\\lambda_{0}^{\\prime} = %snm$" % str(round(popt3[0],2)))
#plt.plot(λs_fit, model(λs_fit, *popt1), alpha=0.6, color="indianred", label="$\\lambda_{0}^{\\prime} = %snm$" % str(round(popt1[0],2)))

#x1, x2, y1, y2 = λs[4800], λs[6800], 0.115, 1.025
#axins = ax.inset_axes([0.1, 0.65, 0.30, 0.30])
#axins.plot(λs[4800:6800], model(λs[4800:6800], *popt), color="lime")
#axins.set_xlim(x1,x2)
#axins.set_ylim(y1,y2)
#axins.set_xticklabels([])
#axins.set_yticklabels([])
#mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="black", alpha=0.3)

#tparams1, _ = curve_fit(model, mist_trans[0], mist_t, p0=[951.2, 951.2, 0.7, 0.5, 0], maxfev=10000)
#tparams2, _ = curve_fit(model, mist_trans[0], mist_t, p0=[951.2, 951.2, 0.7, 0.5, 1e-6], maxfev=10000)

#tparams[4] = 1e-6

#mist_ref_with_losses = theoretical_reflection_values(tparams1, loss_factor = 0.05)[0]
#mist_ref_with_losses = [float(r) for r in mist_ref_with_losses]
#rparams, _ = curve_fit(model, mist_ref[0], mist_ref[1], p0=[951.2, 951.2, 0.3, 0.5, 0], maxfev=10000)
#rparams, _ = curve_fit(model, mist_ref[0], mist_ref_with_losses, p0=[951.2, 951.2, 0.3, 0.5, 1e-6], maxfev=10000)
#print(rparams[4])

#xfit = np.linspace(mist_ref[0][0], mist_ref[0][-1], 10000) 

#lossy_rs = model(xfit, *rparams)
#lossy_ts = model(xfit, *tparams1)
#lossy_losses = np.abs(1 - model(xfit, *tparams1)-model(xfit, *rparams))

#plt.scatter(mist_trans[0], mist_trans[1], marker=".", color="darkorange", label="$T_{MIST}$")
#plt.scatter(mist_trans[0], mist_t, marker=".", color="royalblue", label="$(1-\\varepsilon) \cdot T_{MIST} + \\varepsilon$")
#plt.scatter(mist_ref[0], mist_ref[1], marker=".", color="firebrick", label="$R_{MIST}$")
#plt.plot(xfit, model(xfit, *rparams), color="firebrick", alpha=1, label="$R_{max} = $%s%%" % str(round(np.max(lossy_rs),2)*1e2))#"fit: $\\lambda_0=$%5.3f, $\\lambda_1=$%5.3f, $r_d=$%5.3f, $\\gamma_{\\lambda}=$%5.3f" % tuple(rparams[:-1])) 
#plt.plot(xfit, model(xfit, *tparams1), color="cornflowerblue", alpha=1, label="$T_{min} = $%s%%" % str(round(np.min(lossy_ts),2)*1e2))#"fit: $\\lambda_0=$%5.3f, $\\lambda_1=$%5.3f, $t_d=$%5.3f, $\\gamma_{\\lambda}=$%5.3f" % tuple(tparams[:-1]))
#plt.plot(xfit, model(xfit, *tparams2), color="royalblue", alpha=0.4, label="fit: $(1-\\varepsilon) \cdot T_{MIST} + \\varepsilon$")#label="$T_{min} = $%s%%" % str(round(np.min(lossy_ts),2)*1e2))#"fit: $\\lambda_0=$%5.3f, $\\lambda_1=$%5.3f, $t_d=$%5.3f, $\\gamma_{\\lambda}=$%5.3f" % tuple(tparams[:-1]))
#plt.plot(xfit, lossy_losses, color="blueviolet", alpha=1, label="$L_{max} = $%s%%" % str(round(np.max(lossy_losses),2)*1e2))#"fit: $\\lambda_0=$%5.3f, $\\lambda_1=$%5.3f, $t_d=$%5.3f, $\\gamma_{\\lambda}=$%5.3f" % tuple(tparams[:-1]))


#print("λ0 = ", tparams2[0], "λ1 = ", tparams2[1], "td = ", tparams2[2], "rd = ", rparams[2], "γλ = ", tparams2[3], "β = ", tparams2[4]) 

#print(tparams)
#print(rparams)

#plt.scatter(losses, lws, color="forestgreen", marker=".", label="$\\delta \\lambda (L)$")

#plt.scatter(λs, bb_ts, color="royalblue", marker=".", label="broadband sim.")
#plt.plot(λs, rs1, color="blueviolet", linestyle="-.", label="grating reflectivity", zorder=1)
#plt.scatter(λs, double_ts, color="forestgreen", marker=".", label="double Fano sim.",zorder=2)
#plt.scatter(λs, single_ts, color="orangered", marker=".", label="single Fano sim.",zorder=3)

#plt.plot(λs_fit, model(λs_fit, *popt), color="royalblue", alpha=0.5, label="$HWHM_{broadband} \\approx$ %spm" % str(round(abs(popt[3])*1e3,3)))
#plt.plot(λs_fit, model(λs_fit, *popt1), color="orangered", alpha=0.5, label="$HWHM_{single} \\approx$ %spm" % str(round(abs(popt1[3])*1e3,3)))
#plt.plot(λs_fit, model(λs_fit, *popt2), color="forestgreen", alpha=0.5, label="$HWHM_{double} \\approx$ %spm" % str(round(abs(popt2[3])*1e3,3)))

#plt.plot(λs, bb_ts2, color="firebrick", linestyle="-", label="$|r|^2 = 0.90$")
#plt.plot(λs, single_ts, color="orangered", linestyle="-", label="single fano cavity")

#ax.plot(λs, double_ts1, color="lime")
#ax.plot(λs, double_ts2, color="magenta", linestyle="--")
#ax.plot(λs, double_ts3, color="cyan", linestyle="-.")
#plt.title("Lossless cavity transmission comparison")
#plt.xlabel("wavelength [nm]")
#plt.xlabel("cavity losses, $L = 2(1 - |r_g|^2 - |t_g|^2)$")
#plt.ylabel("HWHM [pm]")
#plt.ylabel("normalized transmission [arb. u.]")
#plt.ylabel("$|E_{out}|^2/|E_{0,in}|^2$")

data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250211/34um/34l.txt")
PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250211/34um/34l_PI.txt")
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250211/normalization/long_scan.txt")
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250211/normalization/long_scan_PI.txt")
λs = np.linspace(data[:,0][0], data[:,0][-1], 10000)

lmin = 33
l1 = (double_cavity_length(G1, G2, λs, lmin=lmin, losses=True) + double_cavity_length(G2, G1, λs, lmin=lmin, losses=True))/2
l2 = double_cavity_length(G1, G2, λs, lmin=lmin, losses=True) 
l3 = double_cavity_length(G2, G1, λs, lmin=lmin, losses=True)
print("l1: ", l1*1e-3)
print("l2: ", l2*1e-3)
print("l3: ", l3*1e-3)

G_ts1 = dual_fano_transmission(G1, G2, l1, λs, intracavity=False, losses=True)
G_ts2 = dual_fano_transmission(G1, G2, l2, λs, intracavity=False, losses=True)
G_ts3 = dual_fano_transmission(G1, G2, l3, λs, intracavity=False, losses=True)

data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])]


plt.plot(λs, G_ts1, color="forestgreen", linestyle="--", label="$l = (l_{G1} + l_{G2})/2$")
plt.plot(λs, G_ts2, color="orangered", linestyle="-.", label="$l = l_{G1}$")
plt.plot(λs, G_ts3, color="cornflowerblue", linestyle="-.", label="$l = l_{G2}$")
plt.scatter(data[:,0], data[:,1], marker=".", color="maroon", label="data", zorder=4)

plt.xlabel("wavelength [nm]", fontsize=28)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
#plt.ylabel("norm. ref./trans. [arb. u.]", fontsize=28)
plt.ylabel("norm. trans. [arb. u.]", fontsize=28)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3)
plt.locator_params(axis='x', tight=True, nbins=8)
plt.show()

#Δs = [0, 0.2, 0.5, 0.8, 1.2]

#detuning_plot(Δs, params1, λs, intracavity=True, losses=False, lmin=30)

#l1 = double_cavity_length(params1, params2, λs, lmin=29.9, losses=False)
#l2 = double_cavity_length(params2, params1, λs, lmin=29.9, losses=False)
#ls = np.linspace(l1,l2,5)
#cavity_length_plot(ls, params1, params2, λs, intracavity=False, losses=False, zoom=False)
#dual_fano_transmission_plot(params1, params2, double_length, λs, intracavity=False, losses=False, zoom=True, grating_trans=True)

#loss_list = [0.12, 0.06, 0.03, 0.01, 0.0]
#loss_factor_scan(params1, loss_list, λs, lmin=30)