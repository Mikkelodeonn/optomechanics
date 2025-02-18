import numpy as np
import matplotlib.pyplot as plt
from fano_class import fano
from scipy.optimize import fsolve
import matplotlib.ticker as ticker

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list, losses=True, loss_factor=0.03):
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

M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")
λ0_1, λ1_1, td_1, γ_1, α_1 = M3.lossy_fit([952,952,0.6,1,0.1])
λ0_2, λ1_2, td_2, γ_2, α_2 = M5.lossy_fit([952,952,0.6,1,0.1])

λ_asymmetry_1 = λ1_1-λ0_1
λ_asymmetry_2 = λ1_2-λ0_2
λ0_1 = 951.535 
λ1_1 = 951.535 + λ_asymmetry_1
λ0_2 = 951.880
λ1_2 = 951.880 + λ_asymmetry_2

λs = np.linspace(M3.data[:,0][0], M3.data[:,0][-1], 1000)

M3_params = [λ0_1, λ1_1, td_1, γ_1, α_1]
M5_params = [λ0_2, λ1_2, td_2, γ_2, α_2]


rs_M3 = theoretical_reflection_values(M3_params, losses=True, loss_factor=0.05)[0]
rs_M5 = theoretical_reflection_values(M5_params, losses=True, loss_factor=0.05)[0]
ts_M3 = model(λs, *M3_params)
ts_M5 = model(λs, *M5_params)

idx = int((list(ts_M3).index(np.min(ts_M3)) + list(ts_M5).index(np.min(ts_M5)))/2)

rt_M3 = rs_M3[idx]
rt_M5 = rs_M5[idx]
tt_M3 = ts_M3[idx]
tt_M5 = ts_M5[idx]
print("ref. of M3 at trans. wavelength:   ", rt_M3[0])
print("ref. of M5 at trans. wavelength:   ", rt_M5[0])
print("trans. of M3 at trans. wavelength: ", tt_M3)
print("trans. of M5 at trans. wavelength: ", tt_M5)

## resonance wavelength [nm -> m]
λres = (λ0_1*1e-9 + λ0_2*1e-9)/2 #955.572e-9
## length of cavity [μm -> m]
l = np.linspace(10,300,10000)*1e-6
## losses in cavity
#L = 0.15
L = (1 - rt_M3 - tt_M3) + (1 - rt_M5 - tt_M5)
print("cavity losses at trans. wavelength:", L[0])
## width of guided mode resonance [nm -> m]
γλ = (γ_1*1e-9 + γ_2*1e-9)/2#0.485*1e-9
## direct (off-resonance) reflectivity (from norm. trans/ref fit)
rd = np.sqrt(0.57)*np.sqrt(0.575)#np.sqrt(((0.57+0.575)/2))#np.sqrt(0.576)
#print(rd)
## Grating transmission at resonance
Tg = tt_M3#0.049
## Broadband mirror transmission at resonance
Tm = tt_M5#0.049

def lw_mirror(l: int, λres: float, L: float, Tg: float, Tm: float): 
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float): 
    δγc = ((λres**2)/(8*np.pi*l)) * (Tg + Tm + L) 
    δγg = ((γλ/(2*(1-rd)))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

def double_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float):
    δγc = ((λres**2)/(8*np.pi*l)) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) * 0.5
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

#for length in lengths:
#    linewidth = 2*lw_fano(length,λres,L,γλ,rd,Tg,Tm)
#    print("length of fano cavity: ", round(length*1e6,1), "μm", " -> ", "theoretical linewidth: ", round(linewidth*1e12,1), "pm")

lengths = np.array([21, 34, 43, 59, 129, 238, 90, 70, 60, 55])*1e-6
lws = np.array([148.169, 77.852, 96.458, 90.403, 61.248, 48.223, 70.428, 66.956, 79.968, 66.54])*1e-12
lw_errs = np.array([10.160799928638458, 6.134863876528573, 24.388186270739908, 7.375280567851888, 5.511886232010013, 
                    5.047405715383159, 6.409594508045273, 6.4623500579952555, 7.026351242285626, 4.154976118278984])*1e-12

ls_0207 = np.array([21, 43, 59, 129, 238])*1e-6
ls_0211 = np.array([34])*1e-6
ls_0218 = np.array([90, 70, 60, 55])*1e-6

lws_0207 = np.array([148.169, 96.458, 90.403, 61.248, 48.223])*1e-12
lws_0211 = np.array([77.852])*1e-12
lws_0218 = np.array([70.428, 66.956, 79.968, 66.54])*1e-12

err_0207 = np.array([10.160799928638458, 24.388186270739908, 7.375280567851888, 5.511886232010013, 5.047405715383159])*1e-12
err_0211 = np.array([6.134863876528573])*1e-12
err_0218 = np.array([6.409594508045273, 6.4623500579952555, 7.026351242285626, 4.154976118278984])*1e-12

sim_ls = np.array([21, 34, 43, 59, 129, 238, 90, 70, 60, 55])*1e-6
sim_lws = np.array([80.911, 75.736, 72.39, 67.117, 50.241, 35.681, 58.109, 62.402, 66.374, 68.336])*1e-12
sim_lw_errs = np.array([0.0005121934467886775, 0.0004377799130283023, 0.0003946381293104754, 0.0003333141561310218, 
                        0.00017926167187670345, 8.649780122407183e-05, 0.0002440424421446865, 0.00029433476802095433, 
                        0.00032524791728291556, 0.000346816907429512])*1e-12

### NOTE: all errors are found as errors of the fit only! ###

plt.figure(figsize=(10,6))

plt.plot(l*1e6,lw_mirror(l,λres,L,Tg,Tm)*1e12, label="broadband cavity")
#plt.plot(l*1e6, double_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="symmetric double fano")
plt.plot(l*1e6,lw_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="single fano cavity")
plt.plot(l*1e6,double_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="double fano cavity")
#plt.errorbar(lengths*1e6,lws*1e12, lw_errs*1e12, fmt=".", capsize=3, color="cornflowerblue", label="HWHM (measured)")
plt.errorbar(ls_0207*1e6,lws_0207*1e12, err_0207*1e12, fmt=".", capsize=3, color="cyan", label="HWHM (measured on 7/2)")
plt.errorbar(ls_0211*1e6,lws_0211*1e12, err_0211*1e12, fmt=".", capsize=3, color="orange", label="HWHM (measured on 11/2)")
plt.errorbar(ls_0218*1e6,lws_0218*1e12, err_0218*1e12, fmt=".", capsize=3, color="limegreen", label="HWHM (measured on 18/2)")

plt.errorbar(sim_ls*1e6, sim_lws*1e12, sim_lw_errs*1e12, fmt=".", capsize=3, color="firebrick", label="HWHM (simulated)")
#plt.plot(λs, rs_M3, "ro")
#plt.plot(λs, rs_M5, "bo")
plt.title("HWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("HWHM [pm]")
plt.xscale("log")
plt.yscale("log")
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.ticklabel_format(style='plain', axis="both")
plt.legend()
plt.show()



