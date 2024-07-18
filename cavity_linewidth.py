import numpy as np
import matplotlib.pyplot as plt

## resonance wavelength [nm]
λres = 951.627e-9
## length of cavity [μm]
l = np.linspace(5e-6,1200e-6,1000000)
## losses in cavity
L = 0.04e-6
## width of guided mode resonance [nm]
γλ = 0.475e-9
## direct reflectivity
rd = 0.3
## Grating transmission at resonance
Tg = 0.029
## Broadband mirror transmission at resonance
Tm =  0.04

def lw_mirror(l: int, λres: float, L: float, Tg: float, Tm: float):
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float):
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg))
    return δγ

print(lw_fano(5e-6,λres,L,γλ,rd,Tg,Tm))

plt.figure(figsize=(10,6))

plt.plot(l*1e6,lw_mirror(l,λres,L,Tg,Tm)*1e12, label="broadband mirror")
plt.plot(l*1e6,lw_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="fano mirror")
plt.title("HWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("HWHM [pm]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()