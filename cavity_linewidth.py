import numpy as np
import matplotlib.pyplot as plt

## resonance wavelength [nm]
λres = 951.627
## length of cavity [μm -> nm]
l = np.linspace(5e3,1200e3,1000000)
## losses in cavity
L = 4
## width of guided mode resonance [nm]
γλ = 0.475
## direct (off-resonance) reflectivity
rd = 0.4
## Grating transmission at resonance
Tg = 2.8
## Broadband mirror transmission at resonance
Tm = 4

def lw_mirror(l: int, λres: float, L: float, Tg: float, Tm: float):
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, Tg: float, Tm: float):
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg))
    return δγ

lengths = [5,10,50,100,500,1000]
for length in lengths:
    linewidth = 2*lw_fano(length*1e3,λres,L,γλ,rd,Tg,Tm)
    print("length of cavity: ", length, " -> ", "theoretical linewidth: ", linewidth)

plt.figure(figsize=(10,6))

plt.plot(l*1e-3,2*lw_mirror(l,λres,L,Tg,Tm), label="broadband mirror")
plt.plot(l*1e-3,2*lw_fano(l,λres,L,γλ,rd,Tg,Tm), label="fano mirror")
plt.title("HWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("HWHM [pm]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

