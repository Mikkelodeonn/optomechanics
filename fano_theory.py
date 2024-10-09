from fano_class import fano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

M1 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M1/400_M1 trans.txt")
M2 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M2/400_M2 trans.txt")
M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M4 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M4/400_M4 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

params1 = M3.lossy_fit([952,952,0.6,1,0.1])
params2 = M3.lossy_fit([952,952,0.6,1,0.1])
#Δ = 0.2
#params2[1] = params2[1] + Δ
#params2[0] = params2[0] + Δ

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

λs_range = np.linspace(950, 953, 1000)
#λs_range = np.linspace(949.9, 950.1, 100)

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list):
    λ0s, λ1s, tds, γλs, βs = params
    γs = 2*np.pi / λ1s**2 * γλs
    a = tds * ((2*np.pi / λ1s) - (2*np.pi / λ0s) - 1j*γs)
    xas = np.real(a)
    yas = np.imag(a)

    L = 0.06
    c_squared = L * (γs**2 + (2*np.pi/λ0s - 2*np.pi/λ1s)**2)

    rds = np.sqrt(1 - tds**2)
    xbs = -(xas * tds / rds)

    def equations(vars):
        yb = vars
        return xas**2 + yas**2 + xbs**2 + yb**2 + 2 * γs * rds * yb + 2 * γs * tds * yas + c_squared
    yb_initial_guess = 0.5
    ybs = fsolve(equations,yb_initial_guess)

    r = []
    for λ_val in λs_range:
        r_val = rds + (xbs + 1j * ybs) / (2 * np.pi / λ_val - 2 * np.pi / λ1s+ 1j * γs)
        r.append(r_val)
    r = np.array(r)
    reflectivity_values = np.abs(r)**2
    complex_reflectivity_amplitudes = r

    return (reflectivity_values, complex_reflectivity_amplitudes)

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

    ls = np.linspace(1,1000,10000)

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

def double_cavity_length(params1: list, params2: list):
    λs = np.linspace(900, 1100, 10000) 

    r1 = theoretical_reflection_values(params1)[1]
    t1 = np.sqrt(model(λs, *params1))
    r2 = theoretical_reflection_values(params2)[1]
    t2 = np.sqrt(model(λs, *params2))

    lengths = []
    Ts = []

    ls = np.linspace(1,1000,10000)

    for l in ls:
        tg1 = np.min(t1)
        rg1 = np.max(r1)
        tg2 = np.min(t2)
        rg2 = np.max(r2)
        λ = params1[0]
        t = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*l)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*l)))**2
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
        Ts.append(float(T))

    return (λs,Ts)

def fano_cavity_transmission_plot(params: list):
    λs, Ts = fano_cavity_transmission(params)
    plt.figure(figsize=(10,6))
    plt.plot(λs, Ts)
    plt.title("Single fano cavity transmission as function of wavelength (l = %sμm)" % str(round(length,2)))
    plt.xlabel("Wavelength [nm]") 
    plt.ylabel("Intensity [arb.u.]")
    plt.show()

def dual_fano_transmission(params1: list, params2: list, length: float):
    
    reflection_values1 = theoretical_reflection_values(params1)[1]
    transmission_values1 = np.sqrt(model(λs_range, *params1))
    reflection_values2 = theoretical_reflection_values(params2)[1]
    transmission_values2 = np.sqrt(model(λs_range, *params2))

    def cavity_transmission(λ, rg1, tg1, rg2, tg2, length):
        T = np.abs(tg1*tg2*np.exp(1j*(2*np.pi/λ)*length)/(1-rg1*rg2*np.exp(2j*(2*np.pi/λ)*length)))**2
        return T 
    λs = np.linspace(λs_range.min(), λs_range.max(), len(reflection_values1))
    Ts = []
    for i in range(len(λs)):
        T = cavity_transmission(λs[i], reflection_values1[i], transmission_values1[i], reflection_values2[i], transmission_values2[i], length)
        Ts.append(float(T))

    return (λs, Ts)

def dual_fano_transmission_plot(params1: list, params2: list, length: float):
    λs, Ts =  dual_fano_transmission(params1, params2, length)
    plt.figure(figsize=(10,6))
    plt.title("Dual fano cavity transmission as a function of wavelength (l = %sμm)" % str(round(length,2)))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
    plt.plot(λs, Ts)
    plt.show()

def detuning_plot(Δs: list): ## plots dual fano cavity transmission for different values for the detuning
    plt.figure(figsize=(10,6))
    grating1 = [950, 950, 0.81, 0.48, 9e-7]
    for Δ in Δs:
        grating2 = [950+Δ, 950+Δ, 0.81, 0.48, 9e-7]
        length = double_cavity_length(grating1, grating2)
        λs, Ts =  dual_fano_transmission(grating1, grating2, length)
        plt.plot(λs, Ts, label="Δ=%snm" %(Δ))

    plt.title("Dual fano cavity transmission as a function of wavelength")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb.u.]")
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

def l_vs_λ_cmaps(params1: list, params2: list): 
    params2[1] += 1
    params2[0] += 1
    Δs = 0.2
    rows = 3
    columns = 3
    Δ_label = 1
    fig, ax = plt.subplots(rows,columns, figsize=(18,8))
    for i in range(rows):
        for j in range(columns):
            params2[1] -= Δs
            params2[0] -= Δs
            Δ_label -= Δs
            ls = np.linspace(double_cavity_length(params1, params2)-1, double_cavity_length(params2, params1)+1, 20)
            Ts = []
            λs = 0
            for l in ls:
                λ, T = dual_fano_transmission(params1, params2, l)
                Ts.append(T)
                λs = λ

            cmap = np.zeros([len(Ts),len(Ts[0])])

            for h in range(len(Ts)):
                for k in range(len(Ts[h])):
                    cmap[h,k] = Ts[h][k] 
            
            l_labels = [round(l,2) for l in ls]
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
    fig.text(0.5, 0.93, 'Double fano transmission for different values of Δ', ha='center', va='center', fontsize=16) 
    fig.text(0.5, 0.06, 'Wavelength [nm]', ha='center', va='center', fontsize=10)
    fig.text(0.08, 0.5, 'Cavity length [μm]', ha='center', va='center', fontsize=10, rotation="vertical")
    plt.show()

def double_fano_cmap(params1: list, params2: list):

    plt.figure(figsize=(10,6))

    ls = np.linspace(double_cavity_length(params1, params2)-1, double_cavity_length(params2, params1)+1,20)
    Ts = []
    λs = 0
    for l in ls:
        λ, T = dual_fano_transmission(params1, params2, l)
        Ts.append(T)
        λs = λ

    cmap = np.zeros([len(Ts),len(Ts[0])])

    for h in range(len(Ts)):
        for k in range(len(Ts[h])):
            cmap[h,k] = Ts[h][k] 

    l_labels = [round(l,2) for l in ls]
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

def line_width_double(params1: list, params2: list): 
    length = double_cavity_length(params1, params2)
    λs, Ts =  dual_fano_transmission(params1, params2, length)

    initial_guess = [951.748, 1, 0.2, 1] #[951.6, 951.6, 0.7, 0.05, 1e-7]

    def lorentzian(x, x0, A, γ, t):
        fano = 1 + ((x-x0) / (γ * (1 - t * (x-x0) / γ)))**2 
        #fano = (γ/2) / ((x-x0)**2 + (γ/2)**2) 
        return A/fano 

    popt, pcov = curve_fit(lorentzian, λs, Ts, p0=initial_guess, maxfev=10000)

    #print("popt: ", popt)

    FWHM = np.abs(2*popt[2])

    print("FWHM: ", round(np.abs(FWHM)*1e3,2), "pm")

    return FWHM*1e3

def line_width_single(params1: list): ## change this !
    λs, Ts =  fano_cavity_transmission(params1)

    initial_guess = [951.748, 1, 0.1, 1] #[951.6, 951.6, 0.7, 0.05, 1e-7]

    def lorentzian(x, x0, A, γ, t):
        fano = 1 + ((x-x0) / (γ * (1 - t * (x-x0) / γ)))**2 
        return A/fano 

    popt, pcov = curve_fit(lorentzian, λs, Ts, p0=initial_guess, maxfev=10000)

    print("popt: ", popt)

    FWHM = 2*popt[2]

    print("length: ", length)
    print("FWHM: ", FWHM)

    plt.figure(figsize=(10,6))
    plt.plot(λs, lorentzian(λs, *popt))
    plt.plot(λs, Ts, 'r.')
    plt.show()

    return FWHM

def line_width_comparison(params1: list, params2: list, length: float): 
    λ1, T1 =  fano_cavity_transmission(params1)
    λ2, T2 = dual_fano_transmission(params1, params2, length)

    initial_guess = [950, 1, 0.1, 1] #[951.6, 951.6, 0.7, 0.05, 1e-7]

    def lorentzian(x, x0, A, γ, t):
        fano = 1 + ((x-x0) / (γ * (1 - t * (x-x0) / γ)))**2 
        return A/fano 

    popt1, pcov1 = curve_fit(lorentzian, λ1, T1, p0=initial_guess, maxfev=10000)
    popt2, pcov2 = curve_fit(lorentzian, λ2, T2, p0=initial_guess, maxfev=10000)

    FWHM_single = 2*popt1[2]
    FWHM_double = 2*popt2[2]

    plt.figure(figsize=(10,6))
    plt.title("Double vs single fano comparison (identical arb. gratings)")
    plt.plot(λ1, T1, '.', color="cornflowerblue", alpha=0.5, label="single fano simulation")
    plt.plot(λ2, T2, 'g.', alpha=0.5, label="double fano simulation")
    plt.plot(λ1, lorentzian(λ1, *popt1), label="single fano fit, FWHM: %spm" %(str(round(FWHM_single*1e3,2))), color="orange")
    plt.plot(λ2, lorentzian(λ2, *popt2), label="double fano fit, FWHM: %spm" %(str(round(FWHM_double*1e3,2))))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [arb. u.]")
    plt.legend()
    plt.show()

    return FWHM


#### double fano transmission as a function of detuning ####

#Δs = np.array([0.01, 0.04, 0.08, 0.12, 0.20, 0.30]) # detuning in nm  
#detuning_plot(Δs)

#### Heat maps of cavity transmission as a function of wavelength and cavity length ####

#l_vs_λ_cmaps(params1,params2)
#double_fano_cmap(params1, params2)


#### Double/single fano cavity transmission plots ####

#length = resonant_cavity_length(params1)
#fano_cavity_transmission_plot(params1)

#length = double_cavity_length(params1, params2)
#dual_fano_transmission_plot(params1, params2, length)


#### for arbitrary line width comparison of the single and double fano models ####

#grating1 = [950, 950, 0.81, 0.48, 9e-7]
#grating2 = grating1
#line_width_comparison(grating1, grating2, double_cavity_length(grating1, grating2))


#### plotting the calculated reflection/transmission values ####

#plt.figure(figsize=(10,6))
#rs = theoretical_reflection_values(params1)[1]
#ts = model(λs_range, *params1)
#plt.plot(λs_range, rs)
#plt.plot(λs_range, ts)
#plt.show()




