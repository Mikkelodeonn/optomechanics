import numpy as np
import matplotlib.pyplot as plt

## resonance wavelength [nm -> m]
λres = 951.7e-9 #955.572e-9
## length of cavity [μm -> m]
l = np.linspace(10,300,10000)*1e-6
## losses in cavity
L = 0.18
## width of guided mode resonance [nm -> m]
γλ = 0.485*1e-9
## direct (off-resonance) reflectivity (from norm. trans/ref fit)
rd = np.sqrt(0.576)
## Grating transmission at resonance
Tg = 0.055
## Broadband mirror transmission at resonance
Tm = 0.055

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

lengths = np.array([21,34,43,59,129,238])*1e-6
#for length in lengths:
#    linewidth = 2*lw_fano(length,λres,L,γλ,rd,Tg,Tm)
#    print("length of fano cavity: ", round(length*1e6,1), "μm", " -> ", "theoretical linewidth: ", round(linewidth*1e12,1), "pm")

lws = np.array([291,170,187,180,129,96])*1e-12 # line widths

plt.figure(figsize=(10,6))

plt.plot(l*1e6,2*lw_mirror(l,λres,L,Tg,Tm)*1e12, label="broadband cavity")
plt.plot(l*1e6,2*lw_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="single fano cavity")
plt.plot(l*1e6,2*double_fano(l,λres,L,γλ,rd,Tg,Tm)*1e12, label="double fano cavity")
plt.plot(lengths*1e6,lws*1e12, "ro", label="measured linewidths (double fano)")
plt.title("FWHM as a function of cavity length")
plt.xlabel("Cavity length [μm]")
plt.ylabel("FWHM [pm]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()



