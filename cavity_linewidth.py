import numpy as np
import matplotlib.pyplot as plt
from fano_class import fano
from scipy.optimize import fsolve
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from statistics import stdev
from statistics import mean

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list, λs: np.array, losses=True, loss_factor=0.04):
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


M3 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/grating trans. spectra/M3/M3_trans.txt")
M5 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/grating trans. spectra/M5/M5_trans_630.txt")
M3_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/grating trans. spectra/M3/M3_trans_PI.txt")
M5_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/grating trans. spectra/M5/M5_trans_630_PI.txt")
M3_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/normalization/grating_trans.txt")
M5_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/normalization/grating_trans.txt")
M3_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/normalization/grating_trans_PI.txt")
M5_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/normalization/grating_trans_PI.txt")

M3[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M3[:,1], M3_PI[:,1], M3_norm[:,1], M3_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
M5[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5[:,1], M5_PI[:,1], M5_norm[:,1], M5_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

λs = np.linspace(M3[:,0][0], M3[:,0][-1], 1000)

p0 = [952,952,0.6,1,0.1]
params1, pcov1 = curve_fit(model, M3[:,0], M3[:,1], p0=p0)
params2, pcov2 = curve_fit(model, M5[:,0], M5[:,1], p0=p0)

#M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
#M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

#λs = np.linspace(M3.data[:,0][0], M3.data[:,0][-1], 1000)

#λ0_1, λ1_1, td_1, γ_1, α_1 = M3.lossy_fit([952,952,0.6,1,0.1])
#λ0_2, λ1_2, td_2, γ_2, α_2 = M5.lossy_fit([952,952,0.6,1,0.1])

#params1 = λ0_1, λ1_1, td_1, γ_1, α_1
#params2 = λ0_2, λ1_2, td_2, γ_2, α_2

λ0_1, λ1_1, td_1, γ_1, α_1 = params1
λ0_2, λ1_2, td_2, γ_2, α_2 = params2

print("λ0s: ", λ0_1, λ0_2) 
print("λ1s: ", λ1_1, λ1_2)
print("tds: ", td_1, td_2)
print("γs:  ", γ_1, γ_2)
print("αs:  ", α_1, α_2)

λt = np.array([0.9*λ0_1 + 0.1*λ0_2]) 

t_M3_trans = model(λt, *params1)
t_M5_trans = model(λt, *params2)

r_M3_trans = theoretical_reflection_values(params1, λt)[0][0]
r_M5_trans = theoretical_reflection_values(params2, λt)[0][0]

r1s = theoretical_reflection_values(params1, λs)[0]
r2s = theoretical_reflection_values(params2, λs)[0]
r1s = [float(r) for r in r1s]
r2s = [float(r) for r in r2s]

rparams1,_ = curve_fit(model, λs, r1s, p0=p0)
rparams2,_ = curve_fit(model, λs, r2s, p0=p0)

## resonance wavelength [nm -> m]
λres = 0.9*λ0_1*1e-9 + 0.1*λ0_2*1e-9
print("resonant wavelength: ", λres)
## length of cavity [μm -> m]
l = np.linspace(15,800,10000)*1e-6
## losses in cavity
Ls = (1 - r_M3_trans) + (1 - r_M5_trans)
print("cavity losses at trans. wavelength:", Ls)
## width of guided mode resonance [nm -> m]
γλ = (γ_1*1e-9 + γ_2*1e-9)/2
## direct (off-resonance) reflectivity (from norm. trans/ref fit)
r1 = rparams1[2]
r2 = rparams2[2]
rd = (r1 + r2 - 2*r1*r2)**2 / (1 - r1*r2)**2 ## the minimum reflectivity is assumed to be the case for the direct/off-resonance case.
print("rd: ", rd)
#print(rd)
## Grating transmission at resonance
Tg = t_M3_trans#0.049
## Broadband mirror transmission at resonance
Tm = t_M5_trans#0.049
print("Tg: ", Tg)
print("Tm: ", Tm)

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

#lengths = np.array([21, 34, 43, 59, 129, 238, 90, 70, 60, 55])*1e-6
#lws = np.array([148.169, 77.852, 96.458, 90.403, 61.248, 48.223, 70.428, 66.956, 79.968, 66.54])*1e-12
#lw_errs = np.array([10.160799928638458, 6.134863876528573, 24.388186270739908, 7.375280567851888, 5.511886232010013, 
#                    5.047405715383159, 6.409594508045273, 6.4623500579952555, 7.026351242285626, 4.154976118278984])*1e-12

#ls_0207 = np.array([21.544, 44.155, 59.943, 130.421, 239.937])*1e-6
#ls_0207_err = np.array([0.134, 0.242, 0.518, 0.970, 0.667])*1e-6

#ls_0211 = np.array([33.283])*1e-6
#ls_0211_err = np.array([0.438])*1e-6

#ls_0218 = np.array([89.441, 64.420, 60.073, 52.909])*1e-6
#ls_0218_err = np.array([0.496, 0.555, 0.342, 0.437])*1e-6

#ls_0220 = np.array([25.369, 41.054, 55.508, 73.002])*1e-6
#ls_0220_err = np.array([0.189, 0.205, 0.347, 0.516])*1e-6

#ls_0225 = np.array([19.930])*1e-6
#ls_0225_err = np.array([0.088])*1e-6

#lws_0207 = np.array([148.169, 96.458, 90.403, 61.248, 48.223])*1e-12
#err_0207 = np.array([10.160799928638458, 24.388186270739908, 7.375280567851888, 5.511886232010013, 5.047405715383159])*1e-12

#lws_0211 = np.array([79.985])*1e-12
#err_0211 = np.array([5.97499144083166])*1e-12

#lws_0218 = np.array([70.428, 66.956, 79.968, 66.54])*1e-12
#err_0218 = np.array([6.409594508045273, 6.4623500579952555, 7.026351242285626, 4.154976118278984])*1e-12

#lws_0220 = np.array([115.698, 79.858, 79.966, 67.24])*1e-12
#err_0220 = np.array([7.130991486232972, 8.382897672891941, 5.877895384766792, 9.143520307376802])*1e-12

#lws_0225 = np.array([(70.684+82.082)/2])*1e-12
#err_0225 = np.array([(5.771+4.231)/2])*1e-12

#sim_ls = np.array([21, 34, 43, 59, 129, 238, 90, 70, 60, 55, 75, 58, 41, 25])*1e-6
#sim_lws = np.array([80.911, 75.736, 72.39, 67.117, 50.241, 35.681, 58.109, 62.402, 66.374, 68.336, 62.404, 67.81, 73.861, 80.577])*1e-12

#sim_ls = np.array([21.452, 44.292, 59.995, 130.421, 239.867, 33.348, 89.498, 64.278, 60.471, 52.858, 25.260, 41.440, 55.716, 72.848, 20.026])*1e-6
#sim_lws = np.array([81.548, 72.691, 67.455, 50.371, 35.678, 76.742, 58.431, 65.281, 66.436, 68.854, 88.797, 81.862, 76.420, 70.640, 74.150])*1e-12
#sim_lw_errs = np.array([0.378, 0.290, 0.246, 0.131, 0.063, 0.328, 0.179, 0.228, 0.237, 0.257, 0.475, 0.382, 0.329, 0.278, 0.302])*1e-12

### NOTE: only peaks which were succesfully fitted to the general double fano model was used (highly diverging line widths were excluded)

err25 = stdev([43.242,82.505,140.995])
err56 = stdev([92.978,131.930,50.198])
err75 = stdev([129.352,113.418,98.231])
err90 = stdev([118.522,127.537,123.208])
err113 = stdev([121.088,119.276,128.153])
err181 = stdev([75.386,78.303,80.703])
err226 = stdev([109.215,91.712,81.864])
err323 = stdev([60.452,52.197,60.835])
err452 = stdev([55.656,56.452,54.841])
err755 = stdev([23.575,30.555,30.757])

lw25 = mean([43.242,82.505,140.995])
lw56 = mean([92.978,131.930,50.198])
lw75 = mean([129.352,113.418,98.231])
lw90 = mean([118.522,127.537,123.208])
lw113 = mean([121.088,119.276,128.153])
lw181 = mean([75.386,78.303,80.703])
lw226 = mean([109.215,91.712,81.864])
lw323 = mean([60.452,52.197,60.835])
lw452 = mean([55.656,56.452,54.841])
lw755 = mean([23.575,30.555,30.757])

lws_0226 = np.array([lw25,lw56,lw75,lw90,lw113,lw181,lw226,lw323,lw452,lw755])*1e-12
errs_0226 = np.array([err25,err56,err75,err90,err113,err181,err226,err323,err452,err755])*1e-12 ## standard deviation of all measurements (within reason)
ls_0226 = np.array([24.612, 54.105, 67.149, 88.825, 106.709, 155.497, 196.168, 299.106, 412.991, 718.787])*1e-6 ## found from fsr scan and fit

lws_0226_good = np.array([lw25,lw56,lw75,lw181,lw323,lw755])*1e-12
errs_0226_good = np.array([err25, err56, err75, err181, err323, err755])*1e-12 ## standard deviation of all measurements (within reason)
ls_0226_good = np.array([24.612, 54.105, 67.149, 155.497, 299.106, 718.787])*1e-6 ## found from fsr scan and fit

sim_ls = np.array([24.768, 54.269, 68.068, 155.620, 299.795, 718.997])*1e-6
sim_lws = np.array([79.035, 62.312, 56.731, 36.268, 22.806, 10.995])*1e-12

plt.figure(figsize=(10,6))

#plt.plot(l*1e6,lw_mirror(l,λres,Ls,Tg,Tm)*1e12, label="broadband cavity")
#plt.plot(l*1e6,lw_mirror(l,λres2,L2,Tg2,Tm2)*1e12, label="broadband cavity")
#plt.plot(l*1e6,lw_mirror(l,λres3,L3,Tg3,Tm3)*1e12, label="broadband cavity")
plt.plot(l*1e6,lw_mirror(l,λres,Ls,Tg,Tm)*1e12, label="broadband cavity")
plt.errorbar(ls_0226*1e6, lws_0226*1e12,errs_0226*1e12, fmt=".", capsize=3, color="orange", label="HWHM (measured on 26/2)")
#plt.errorbar(ls_0226_good*1e6, lws_0226_good*1e12, errs_0226_good*1e12, fmt=".", color="magenta", capsize=3, label="HWHM (measured on 26/2)")

plt.plot(l*1e6,lw_fano(l,λres,Ls,γλ,rd,Tg,Tm)*1e12, label="single fano cavity")
plt.plot(l*1e6,double_fano(l,λres,Ls,γλ,rd,Tg,Tm)*1e12, label= "double fano cavity")
#plt.plot(l*1e6,double_fano(l,λres,Ls,γλ,rd,Tg,Tg)*1e12, label= "symmetric double fano cavity")
#plt.plot(l*1e6,double_fano(l,λres1,L1,γλ,rd,Tg1,Tm1)*1e12, label= "double fano cavity")
#plt.plot(l*1e6,double_fano(l,λres2,L2,γλ,rd,Tg2,Tm2)*1e12, label = "double fano cavity (adjusted detuning)")
#plt.plot(l*1e6,double_fano(l,λres3,L3,γλ,rd,Tg3,Tm3)*1e12, label = "double fano cavity (adjusted detuning)")
#plt.errorbar(lengths*1e6,lws*1e12, lw_errs*1e12, fmt=".", capsize=3, color="cornflowerblue", label="HWHM (measured)")
#plt.errorbar(ls_0207*1e6, lws_0207*1e12, err_0207*1e12, xerr=ls_0207_err*1e6, fmt=".", capsize=3, color="cyan", label="HWHM (measured on 7/2)")
#plt.errorbar(ls_0211*1e6, lws_0211*1e12, err_0211*1e12, xerr=ls_0211_err*1e6, fmt=".", capsize=3, color="orange", label="HWHM (measured on 11/2)")
#plt.errorbar(ls_0218*1e6, lws_0218*1e12, err_0218*1e12, xerr=ls_0218_err*1e6, fmt=".", capsize=3, color="limegreen", label="HWHM (measured on 18/2)")
#plt.errorbar(ls_0220*1e6, lws_0220*1e12, err_0220*1e12, xerr=ls_0220_err*1e6, fmt=".", capsize=3, color="magenta", label="HWHM (measured on 20/2)")
#plt.errorbar(ls_0225*1e6, lws_0225*1e12, err_0225*1e12, xerr=ls_0225_err*1e6, fmt=".", capsize=3, color="darkblue", label="HWHM (measured on 25/2)")

plt.scatter(sim_ls*1e6, sim_lws*1e12, marker=".", color="limegreen", label="HWHM (simulated)", zorder=7)
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



