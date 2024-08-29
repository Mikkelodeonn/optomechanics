import numpy as np
import matplotlib.pyplot as plt

## resonance wavelength [nm -> m]
λres = 951.930*1e-9
## length of cavity [μm -> m]
l = np.linspace(5,12000,1000000)*1e-6
## losses in cavity
L = 0.12
## width of guided mode resonance [nm -> m]
γλ = 0.483*1e-9
## direct (off-resonance) reflectivity
rd = np.sqrt(0.38)
## Grating transmission at resonance
Tg = 0.024
## Broadband mirror transmission at resonance 
Tm = 0.04

def lw_mirror(l: int, λres: float, L: float, Tg: float, Tm: float):
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float):
    δγc = (λres**2)/(8*np.pi*l) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg))
    return δγ

lengths = np.array([823,724,647,604,566,453,394,201,162,92,81,22,10])*1e-6 # lengths in m
for length in lengths:
    linewidth = 2*lw_fano(length,λres,L,γλ,rd,Tg,Tm)
    print("length of fano cavity: ", round(length*1e6,1), "μm", " -> ", "theoretical linewidth: ", round(linewidth*1e12,1), "pm")

lws = np.array([16,16,66,22,30,40,30,60,80,88,84,200,122])*1e-9 ## linewidths found for the 400um grating

plt.figure(figsize=(10,6))

plt.plot(l*1e6,2*lw_mirror(l,λres,L,Tg,Tm)*1e12, label="broadband cavity")
plt.plot(l*1e6,2*lw_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="fano cavity")
plt.plot(l*1e6,2*lw_fano(l,λres,0.08,γλ,rd,Tg,Tg)*1e12, label="dual fano cavity", ls="--")
plt.plot(lengths*1e6,lws*1e9, "ro", label="measured linewidths")
plt.title("FWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("FWHM [pm]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()



