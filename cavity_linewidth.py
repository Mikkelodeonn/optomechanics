import numpy as np
import matplotlib.pyplot as plt

## resonance wavelength [nm -> m]
λres = 951.032*1e-9
## length of cavity [μm -> m]
l = np.linspace(5,1200,1000000)*1e-6
## losses in cavity
L = 0.04
## width of guided mode resonance [nm -> m]
γλ = 0.475*1e-9
## direct (off-resonance) reflectivity
rd = 0.42 # 40%
## Grating transmission at resonance
Tg = 0.028 # 2.8%
## Broadband mirror transmission at resonance
Tm = 0.04 # 4%

def lw_mirror(l: int, λres: float, L: float, Tg: float, Tm: float):
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float):
    δγc = (λres**2)/(8*np.pi*l) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg))
    return δγ

lengths = np.array([5,10,50,100,500,1000])*1e-6 # lengths in m
for length in lengths:
    linewidth = 2*lw_fano(length,λres,L,γλ,rd,Tg,Tm)
    print("length of cavity: ", round(length*1e6,1), "μm", " -> ", "theoretical linewidth: ", round(linewidth*1e12,1), "pm")

plt.figure(figsize=(10,6))

plt.plot(l*1e6,2*lw_mirror(l,λres,L,Tg,Tm)*1e12, label="broadband mirror")
plt.plot(l*1e6,2*lw_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="fano mirror")
plt.title("HWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("FWHM [pm]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

