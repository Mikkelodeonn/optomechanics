import numpy as np
import matplotlib.pyplot as plt
from fano_class import fano
from scipy.optimize import fsolve

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

#print(λ0_1-λ1_1)
#print(λ0_2-λ1_2)
λ0_1 = 951.535 #551.535
λ1_1 = 951.535 + 0.14
λ0_2 = 951.875
λ1_2 = 951.875 + 0.15

λs = np.linspace(M3.data[:,0][0], M3.data[:,0][-1], 1000)

M3_params = [λ0_1, λ1_1, td_1, γ_1, α_1]
M5_params = [λ0_2, λ1_2, td_2, γ_2, α_2]

rs_M3 = theoretical_reflection_values(M3_params, losses=True, loss_factor=0.03)[0]
rs_M5 = theoretical_reflection_values(M5_params, losses=True, loss_factor=0.03)[0]
ts_M3 = model(λs, *M3_params)
ts_M5 = model(λs, *M5_params)

idx = int((list(rs_M3).index(np.max(rs_M3)) + list(rs_M5).index(np.max(rs_M5)))/2)
#idx = int(list(rs_M3).index(np.max(rs_M3)))
#idx = int(list(rs_M5).index(np.max(rs_M5)))

rt_M3 = rs_M3[idx]
rt_M5 = rs_M5[idx]
tt_M3 = ts_M3[idx]
tt_M5 = ts_M5[idx]
print(rt_M3)
print(rt_M5)
print(tt_M3)
print(tt_M5)

print(idx)

## resonance wavelength [nm -> m]
λres = (λ0_1*1e-9 + λ0_2*1e-9)/2 #955.572e-9
## length of cavity [μm -> m]
l = np.linspace(10,300,10000)*1e-6
## losses in cavity
#L = 0.15
L = (1- rt_M3 - tt_M3) + (1 - rt_M5 - tt_M5)
print("L: ",L)
## width of guided mode resonance [nm -> m]
γλ = (γ_1*1e-9 + γ_2*1e-9)/2#0.485*1e-9
## direct (off-resonance) reflectivity (from norm. trans/ref fit)
rd = np.sqrt(((0.57+0.575)/2))#np.sqrt(0.576)
#print(rd)
## Grating transmission at resonance
Tg = tt_M3#0.049
## Broadband mirror transmission at resonance
Tm = tt_M5#0.049

print(Tg, Tm)

def lw_mirror(l: int, λres: float, L: float, Tg: float, Tm: float): 
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float): 
    δγc = (λres**2)/(8*np.pi*l) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

def double_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float):
    δγc = (λres**2)/(8*np.pi*l) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) * 0.5
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

#lengths = np.array([900])*1e-6 # cavity lengths

#lengths = np.array([21,34,43,59,129,238])*1e-6
#lengths = np.array([20.1,30.18,40.04,59.08,139,238.9])*1e-6
#for length in lengths:
#    linewidth = 2*lw_fano(length,λres,L,γλ,rd,Tg,Tm)
#    print("length of fano cavity: ", round(length*1e6,1), "μm", " -> ", "theoretical linewidth: ", round(linewidth*1e12,1), "pm")

#lws = np.array([291.28,173.5,187.6,180.4,129.2,96.63])*1e-12*0.5 # linewidths in pm

lengths = np.array([21, 34, 43, 59, 129, 238])*1e-6
lws = np.array([148.169, 86.019, 96.282, 86.945, 65.258, 49.974])*1e-12

plt.figure(figsize=(10,6))

plt.plot(l*1e6,lw_mirror(l,λres,0.08,0.049,0.049)*1e12, label="broadband cavity")
plt.plot(l*1e6, double_fano(l,λres,0.08,γλ,rd,0.049,0.049)*1e12, label="symmetric double fano")
plt.plot(l*1e6,lw_fano(l,λres,0.08,γλ,rd,0.049,0.049)*1e12, label="ideal single fano cavity")
plt.plot(l*1e6,double_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="asymmetric double fano cavity")
plt.plot(lengths*1e6,lws*1e12, "ro", label="measured linewidths (double fano)")
#plt.plot(λs, rs_M3)
#plt.plot(λs, rs_M5)
plt.title("HWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("HWHM [pm]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()



