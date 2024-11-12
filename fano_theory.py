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

params1 = M3.lossy_fit([952,952,0.6,1,0.1])
params2 = M5.lossy_fit([952,952,0.6,1,0.1])

## grating parameters -> [λ0, λ1, td, γλ, α]
# λ0 -> resonance wavelength
# λ1 -> guided mode resonance wavelength
# td -> direct transmission coefficient
# γλ -> width of guided mode resonance
# α  -> loss factor 

#λs = np.linspace(951, 952.5, 500)
λs = np.linspace(950, 953, 50)
#λs = np.linspace(910, 980, 10000)
#λs = np.linspace(951.7, 951.85, 200)

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list, losses=True):
    λ0s, λ1s, tds, γλs, βs = params
    γs = 2*np.pi / λ1s**2 * γλs
    a = tds * ((2*np.pi / λ1s) - (2*np.pi / λ0s) + 1j*βs - 1j*γs)
    xas = np.real(a)
    yas = np.imag(a)

    if losses == True:
        L = 0.03
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

    popt_t1, _ = curve_fit(model, λs, t1, p0=[951.8,951.8,0.6,1,1e-7])
    popt_t2, _ = curve_fit(model, λs, t2, p0=[951.8,951.8,0.6,1,1e-7])
    t1_fit = model(λs_fit, *popt_t1)
    t2_fit = model(λs_fit, *popt_t2)
    tidx1 = np.argmin(t1_fit)
    tidx2 = np.argmin(t2_fit)
    tmin1 = t1_fit[tidx1]
    tmin2 = t2_fit[tidx2]

    popt_r1, _ = curve_fit(model, λs, r1, p0=[951.8,951.8,0.4,1,1e-7])
    popt_r2, _ = curve_fit(model, λs, r2, p0=[951.8,951.8,0.4,1,1e-7])
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

def double_cavity_length(params1: list, params2: list, λs: np.array, lmin=50):
    r1 = theoretical_reflection_values(params1, losses=True)[1]
    r2 = theoretical_reflection_values(params2, losses=True)[1]
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

def double_fano_length_scan(params1: list, params2: list, ls: np.array, λs: np.array):
    t1 = [complex(t) for t in np.array(np.sqrt(model(λs, *params1)))]
    t2 = [complex(t) for t in np.array(np.sqrt(model(λs, *params2)))]
    r1 = [complex(r) for r in theoretical_reflection_values(params1, losses=True)[1]]
    r2 = [complex(r) for r in theoretical_reflection_values(params2, losses=True)[1]]

    idx = np.argmin(t1)

    rg1 = r1[idx]; rg2 = r2[idx]
    tg1 = t1[idx]; tg2 = t2[idx]

    λ = np.array([λs[idx]])
    Ts = []
    for l in ls:
        T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
        Ts.append(T)
    
    resonance_length = double_cavity_length(params1, params2, λs, lmin=ls[0]*1e-3)*1e-3

    plt.figure(figsize=(10,6))
    plt.plot(ls*1e-3, Ts, "cornflowerblue")
    plt.title("Double fano cavity transmission as a function of cavity length")
    plt.xlabel("cavity length [μm]")
    plt.ylabel("Transmission [arb. u.]")
    plt.legend(["Resonance cavity length: %sμm" % str(round(resonance_length,3))]) 
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

def fano_cavity_transmission_plot(params: list, length: np.array, λs: np.array, intracavity=False, losses=True):
    Ts = fano_cavity_transmission(params, length, λs, intracavity=intracavity, losses=losses)
    plt.figure(figsize=(10,6))
    plt.plot(λs, Ts)
    plt.title("Single fano cavity transmission as function of wavelength (l = %sμm)" % str(round(length*1e-3,2)))
    plt.xlabel("Wavelength [nm]") 
    plt.ylabel("Intensity [arb.u.]")
    plt.show()

def dual_fano_transmission(params1: list, params2: list, length: float, λs: np.array, intracavity=False, losses=True):
    #print("double fano length: ", length)
    
    reflection_values1 = theoretical_reflection_values(params1, losses=losses)[1]
    transmission_values1 = np.sqrt(model(λs, *params1))
    reflection_values2 = theoretical_reflection_values(params2, losses=losses)[1]
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

def dual_fano_transmission_plot(params1: list, params2: list, length: float, λs: np.array, intracavity=False, losses=True, zoom=False, total_grating_trans=False):
    Ts =  dual_fano_transmission(params1, params2, length, λs, intracavity=intracavity, losses=losses)
    fig, ax = plt.subplots(figsize=(10,6))
    if zoom == True:
        x1, x2, y1, y2 = params1[0]-1, params1[0]+1, 0.05, 0.6
        axins = ax.inset_axes([0.7, 0.65, 0.25, 0.25])
        axins.plot(λs,Ts)
        axins.set_xlim(x1,x2)
        axins.set_ylim(y1,y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        mark_inset(ax, axins, loc1=2, loc2=4, edgecolor="black", alpha=0.3)
    if total_grating_trans ==True:
        tg1 = model(λs, *params1)
        tg2 = model(λs, *params2)
        tg_total = np.abs(tg1*tg2)
        ax.plot(λs, tg_total, "gray", linestyle="--", alpha=0.6, label=r"$|t_{M3} \cdot t_{M5}|$")
    ax.set_title("Double fano cavity transmission as a function of wavelength (l = %sμm)" % str(round(length*1e-3,3)))
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
            linesize = 2
        else:
            linesize = 2
        plt.plot(λs, Ts, color=paint, linestyle=style, linewidth=linesize, label="Δ=%snm" %(round(Δ,2)))

    #plt.title("Double fano transmission for varying detuning Δ (cavity length %s)" %(r"$\rightarrow l_{g,1} \approx 100 \mu m$"))
    plt.title("Double fano transmission for varying detuning Δ (cavity length %s)" %(r"$\rightarrow (l_{g,1} + l_{g,2})/2 \approx 10 \mu m$"))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [arb.u.]")
    plt.legend()
    plt.show()

def cavity_length_plot(ls: list, params1: list, params2: list, λs: np.array, intracavity=False, losses=True):
    plt.figure(figsize=(15,6))
    linestyles = ["-.", "--", "-", "--", "-."]
    colors = ["skyblue","royalblue","forestgreen", "firebrick", "lightcoral"]
    for l, paint, style in zip(ls, colors, linestyles):
        Ts = dual_fano_transmission(params1, params2, l, λs, intracavity=intracavity, losses=losses)
        plt.plot(λs, Ts, color=paint, linestyle=style, linewidth=2, label="cavity length: %sμm" % str(round(l*1e-3,4)))
    plt.title("Double fano transmission for different cavity lengths %s (M3/M5)" %(r"$l_{g,1} \rightarrow l_{g,2}$")) 
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

def double_fano_cmap(params1: list, params2: list, λs: np.array, lmin=50):

    plt.figure(figsize=(10,6))

    ls = np.linspace(double_cavity_length(params1, params2, λs, lmin=lmin), double_cavity_length(params2, params1, λs, lmin=lmin),20)
    Ts = []
    for l in ls:
        T = dual_fano_transmission(params1, params2, l, λs)
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
    FWHM_print = str(round(FWHM,2)) + "+/-" + str(round(err,2))
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


#### double fano transmission as a function of detuning #### 

#Δs = np.linspace(-1.5, 1.5, 5) # low resolution
#Δs = np.linspace(-0.3, 0.3, 5) # high resolution
#Δs = np.linspace(0, 1, 5)
#detuning_plot(Δs, params1, λs, intracavity=False, losses=True, lmin=5)

#### Heat maps of cavity transmission as a function of wavelength and cavity length ####

#l_vs_λ_cmaps(params1, params2, λs, intracavity=False, losses=True, lmin=30)
#double_fano_cmap(params1, params2, λs, lmin=10)


#### Double/single fano cavity transmission plots ####

#length = resonant_cavity_length(params1, λs, lmin=10)
#fano_cavity_transmission_plot(params1, length, λs, intracavity=False, losses=True)

#length = (double_cavity_length(params1, params2, λs, lmin=30) + double_cavity_length(params2, params1, λs, lmin=30))/2
#length = double_cavity_length(params1, params2, λs, lmin=10)
#dual_fano_transmission_plot(params1, params2, length, λs, intracavity=False, losses=True, total_grating_trans=True, zoom=False)

#Δ = 0.1
#lmin=51
#params2[0] += Δ
#params2[1] += Δ
#ls = np.linspace(double_cavity_length(params1,params2,λs,lmin=lmin), double_cavity_length(params2,params1,λs,lmin=lmin), 5)
#cavity_length_plot(ls, params1, params2, λs, intracavity=False, losses=True)


#### for line width comparison of the single and double fano models ####

#grating1 = [951, 951, 0.81, 0.48, 1e-6]
#grating2 = grating1
#lmin = 30
#length = double_cavity_length(params1, params2, λs, lmin=lmin)*0.25 + double_cavity_length(params2, params1, λs, lmin=lmin)*0.75
#print("cavity length: ", length*1e-3)
#line_width_comparison(grating1, grating2, double_cavity_length(grating1, grating2, λs), intracavity=True, losses=False)
#line_width_comparison(params1, params2, double_cavity_length(params1, params2, λs), intracavity=True, losses=True)
#line_width_single(params1, λs)
#line_width_double(params1, params2, λs, length)

#### length scan of the single and double fano cavities

#l=10
#ls = np.linspace(l*1e3, (l+1)*1e3, 1000)
#double_fano_length_scan(params1, params2, ls, λs)
#single_fano_length_scan(params1, ls, λs)


#### plotting the calculated reflection/transmission values ####

#theoretical_reflection_values_plot(params1, λs)
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
#linewidths = [79.8997, 75.7106, 74.117, 76.2582, 79.8249]
#cavity_lengths = [30.025460254602546, 30.021697716977172, 30.01810018100181, 30.014460144601447, 30.010830108301082]
#errors = [0.2903, 0.15, 0.0768, 0.0982, 0.3143]

#plt.errorbar(cavity_lengths, linewidths, errors, fmt="o", color="cornflowerblue", capsize=5)
#plt.title("double Fano linewidth as a function of cavitylength (lengths: %s)" % r"$l_{M3} \rightarrow l_{M5} \approx 30μm$")
#plt.ylabel("FWHM [pm]")
#plt.xlabel("cavity length [μm]")      
#plt.show()




