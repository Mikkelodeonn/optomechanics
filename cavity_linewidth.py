import numpy as np
import matplotlib.pyplot as plt
from fano_class import fano
from scipy.optimize import fsolve
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from statistics import stdev
from statistics import mean


## detunings:
## 07/02 -> M3 = 951.535, M5 = 951.875
## 11/02 -> M3 = 951.535, M5 = 951.800
## 18/02 -> M3 = 951.540, M5 = 951.800
## 20/02 -> M3 = 951.570, M5 = 951.950

## make theoretical prediction such that the graph shows the average of all measurements of params1 and params2 (estimate the above detuning pairs),
## and a shaved area indicates the standard deviation of alle the sets of parameters.

def model(λ, λ0, λ1, td, γλ, β): 
    k = 2*np.pi / λ
    k0 = 2*np.pi / λ0
    k1 = 2*np.pi / λ1
    γ = 2*np.pi / λ1**2 * γλ
    t = td * (k - k0 + 1j * β) / (k - k1 + 1j * γ)
    return np.abs(t)**2

def theoretical_reflection_values(params: list, λs: np.array, losses=True, loss_factor=0.03):
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

def lw_mirror(l: int, λres: float, L: float, r1: float, r2: float): 
    δγc = (λres**2/(8*np.pi*l)) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, L: float, γλ: float, rd: float, r1: float, r2: float): 
    δγc = ((λres**2)/(8*np.pi*l)) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    δγg = ((γλ/(2*(1-rd)))) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

def double_fano(l: int, λres: float, L: float, γλ: float, rd: float, r1: float, r2: float):
    δγc = ((λres**2)/(8*np.pi*l)) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (1 - r1 + 1 - r2)*0.5 #(Tg + Tm + L) * 0.5
    #print(δγg)
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ

def double_fano_losses(l: int, λres: float, L: float, γλ: float, rd: float, t1: float, t2: float):
    δγc = ((λres**2)/(8*np.pi*l)) * (t1 + t2 + L) #(Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (t1 + t2 + L)*0.5 #(Tg + Tm + L) * 0.5
    #print(δγg)
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ

def calc_lws(l, params1, params2, losses=True):
    λ0_1 = params1[0]; γ_1 = params1[3]
    λ0_2 = params2[0]; γ_2 = params2[3]

    λt = np.array([0.5*λ0_1 + 0.5*λ0_2]) 

    t_M3_trans = model(λt, *params1)
    t_M5_trans = model(λt, *params2)

    r_M3_trans = theoretical_reflection_values(params1, λt, losses=losses)[0][0]
    r_M5_trans = theoretical_reflection_values(params2, λt, losses=losses)[0][0]

    r1s = theoretical_reflection_values(params1, λs, losses=losses)[0]
    r2s = theoretical_reflection_values(params2, λs, losses=losses)[0]
    r1s = [float(r) for r in r1s]
    r2s = [float(r) for r in r2s]

    rparams1, _ = curve_fit(model, λs, r1s, p0=p0)
    rparams2, _ = curve_fit(model, λs, r2s, p0=p0)

    ## resonance wavelength [nm -> m]
    λres = 0.5*λ0_1*1e-9 + 0.5*λ0_2*1e-9
    #print("resonant wavelength: ", λres)
    ## length of cavity [μm -> m]
    ## losses in cavity
    Ls = (1 - r_M3_trans) + (1 - r_M5_trans)
    print("cavity losses at trans. wavelength:", Ls)
    ## width of guided mode resonance [nm -> m]
    γλ = (0.5*γ_1*1e-9 + 0.5*γ_2*1e-9)
    print("γ: ", γλ)
    ## direct (off-resonance) reflectivity (from norm. trans/ref fit)
    rd1 = rparams1[2]
    rd2 = rparams2[2]
    rd = (rd1 + rd2)/2
    #rd = (rd1 + rd2 - 2*rd1*rd2)**2 / (1 - rd1*rd2)**2 ## the minimum reflectivity is assumed to be the case for the direct/off-resonance case.
    print("rd: ", rd)
    #print(rd)
    ## Grating transmission at resonance
    Tg = t_M3_trans#0.049
    ## Broadband mirror transmission at resonance
    Tm = t_M5_trans#0.049
    #print("Tg: ", Tg)
    #print("Tm: ", Tm)

    double_fano_lws = double_fano(l,λres,Ls,γλ,rd,r_M3_trans,r_M5_trans)*1e12

    single_fano_lws = lw_fano(l,λres,Ls,γλ,rd,r_M3_trans,r_M5_trans)*1e12

    mirror_lws = lw_mirror(l, λres, Ls, r_M3_trans, r_M5_trans)*1e12

    return [double_fano_lws, single_fano_lws, mirror_lws]

l = np.linspace(15,800,10000)*1e-6

M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

params1_origin = M3.lossy_fit([952,952,0.6,1,0.1])
params2_origin = M5.lossy_fit([952,952,0.6,1,0.1])

M31 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/grating trans. spectra/M3/M3_trans.txt")
M51 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/grating trans. spectra/M5/M5_trans_630.txt")
M31_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/grating trans. spectra/M3/M3_trans_PI.txt")
M51_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/grating trans. spectra/M5/M5_trans_630_PI.txt")
M31_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/normalization/grating_trans.txt")
M51_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/normalization/grating_trans.txt")
M31_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250226/normalization/grating_trans_PI.txt")
M51_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250303/normalization/grating_trans_PI.txt")

M31[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M31[:,1], M31_PI[:,1], M31_norm[:,1], M31_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
M51[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M51[:,1], M51_PI[:,1], M51_norm[:,1], M51_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

M32 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250305/grating trans. spectra/M3_trans.txt")
M52 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250305/grating trans. spectra/M5_trans.txt")
M32_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250305/grating trans. spectra/M3_trans_PI.txt")
M52_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250305/grating trans. spectra/M5_trans_PI.txt")
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250305/normalization/grating_trans.txt")
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250305/normalization/grating_trans_PI.txt")


M3_0326 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M3_trans.txt")
M5_0326 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M5_trans.txt")
M3_0326_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M3_trans_PI.txt")
M5_0326_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/grating trans. spectra/M5_trans_PI.txt")
norm_0326 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/grating_trans.txt")
norm_0326_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/grating_trans_PI.txt")

M32[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M32[:,1], M32_PI[:,1], norm[:,1], norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
M52[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M52[:,1], M52_PI[:,1], norm[:,1], norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity.

M3_0326[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M3_0326[:,1], M3_0326_PI[:,1], norm_0326[:,1], norm_0326_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
M5_0326[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5_0326[:,1], M5_0326_PI[:,1], norm_0326[:,1], norm_0326_PI[:,1])] ## norm. with respect to trans. w/o a cavity.

λs = np.linspace(M31[:,0][0], M31[:,0][-1], 1000)

p0 = [952,952,0.6,1,0.1]
params1_0226, pcov1 = curve_fit(model, M31[:,0], M31[:,1], p0=p0)
params2_0226, pcov2 = curve_fit(model, M51[:,0], M51[:,1], p0=p0)

params1_0305, pcov1 = curve_fit(model, M32[:,0], M32[:,1], p0=p0)
params2_0305, pcov2 = curve_fit(model, M52[:,0], M52[:,1], p0=p0)

params1_0326, pcov1_0326 = curve_fit(model, M3_0326[:,0], M3_0326[:,1], p0=p0)
params2_0326, pcov2_0326 = curve_fit(model, M5_0326[:,0], M5_0326[:,1], p0=p0)

params1_0326_errs = np.sqrt(np.diag(pcov1_0326))
params2_0326_errs = np.sqrt(np.diag(pcov2_0326))


#print(params1_0226-params1_0305)
#print(params2_0226-params2_0305)

asym1 = np.abs(params1_0226[1] - params1_0226[0])
asym2 = np.abs(params2_0226[1] - params2_0226[0])

## 07/02 -> M3 = 951.535, M5 = 951.875
## 11/02 -> M3 = 951.535, M5 = 951.800
## 18/02 -> M3 = 951.540, M5 = 951.800
## 20/02 -> M3 = 951.570, M5 = 951.950

p1_0702 = params1_0226.copy(); p1_0702[0] = 951.535; p1_0702[1] = 951.535 + asym1
p1_1102 = params1_0226.copy(); p1_1102[0] = 951.535; p1_1102[1] = 951.535 + asym1
p1_1802 = params1_0226.copy(); p1_1802[0] = 951.540; p1_1802[1] = 951.540 + asym1
p1_2002 = params1_0226.copy(); p1_2002[0] = 951.570; p1_2002[1] = 951.570 + asym1

p2_0702 = params2_0226.copy(); p2_0702[0] = 951.875; p2_0702[1] = 951.875 + asym2 
p2_1102 = params2_0226.copy(); p2_1102[0] = 951.800; p2_1102[1] = 951.800 + asym2 
p2_1802 = params2_0226.copy(); p2_1802[0] = 951.800; p2_1802[1] = 951.800 + asym2 
p2_2002 = params2_0226.copy(); p2_2002[0] = 951.950; p2_2002[1] = 951.950 + asym2 

p1_errs = []
p2_errs = []

for i in range(len(params1_0226)):
    p1_err = stdev([params1_origin[i], params1_0226[i], params1_0305[i], p1_0702[i], p1_1102[i], p1_1802[i], p1_2002[i]], params1_0326[i])
    p2_err = stdev([params1_origin[i], params2_0226[i], params2_0305[i], p2_0702[i], p2_1102[i], p2_1802[i], p2_2002[i]], params2_0326[i])
    p1_errs.append(p1_err)
    p2_errs.append(p2_err)


ls_0207 = np.array([21.544, 44.155, 59.943, 130.421, 239.937])*1e-6
ls_0207_err = np.array([0.134, 0.242, 0.518, 0.970, 0.667])*1e-6

ls_0211 = np.array([33.283])*1e-6
ls_0211_err = np.array([0.438])*1e-6

ls_0218 = np.array([89.441, 64.420, 60.073, 52.909])*1e-6
ls_0218_err = np.array([0.496, 0.555, 0.342, 0.437])*1e-6

ls_0220 = np.array([25.369, 41.054, 55.508, 73.002])*1e-6
ls_0220_err = np.array([0.189, 0.205, 0.347, 0.516])*1e-6

ls_0314 = np.array([92.162, 143.181])*1e-6

lws_0207 = np.array([139.644, 96.458, 90.403, 61.248, 48.223])*1e-12
err_0207 = np.array([5.019, 24.388186270739908, 7.375280567851888, 5.511886232010013, 5.047405715383159])*1e-12

lws_0211 = np.array([79.985])*1e-12
err_0211 = np.array([5.97499144083166])*1e-12

lws_0218 = np.array([70.428, 66.956, 79.968, 66.54])*1e-12
err_0218 = np.array([6.409594508045273, 6.4623500579952555, 7.026351242285626, 4.154976118278984])*1e-12

lws_0220 = np.array([115.698, 79.858, 79.966, 67.24])*1e-12
err_0220 = np.array([7.130991486232972, 8.382897672891941, 5.877895384766792, 9.143520307376802])*1e-12

lws_0226 = np.array([82.505])*1e-12
err_0226 = np.array([30.503])*1e-12


### 20250326 ###

ls_0326 = np.array([21.390, 32.736, 51.396, 81.291, 245.305, 321.768])*1e-6

lws21 = np.array([77.869, 55.452, 55.252, 73.609, 73.940, 54.196, 64.092, 64.803, 67.450, 68.652, 62.498, 62.435])
lws33 = np.array([46.428, 65.247, 47.547, 79.556, 76.220, 83.444, 65.594, 82.781, 57.836, 65.532, 84.365, 80.165, 49.984, 69.228, 66.839])
lws53 = np.array([69.904, 56.648, 63.267, 56.318, 41.177, 62.520, 55.533])
lws83 = np.array([48.060, 67.618, 73.128, 71.593, 48.218, 64.344, 67.277])
lws251 = np.array([37.840, 46.683, 44.496])
lws323 = np.array([39.196, 52.158, 56.468])
lws453 = np.array([36.676, 36.491, 39.229])

lw21 = mean(lws21)
lw33 = mean(lws33)
lw53 = mean(lws53)
lw83 = mean(lws83)
lw251 = mean(lws251)
lw323 = mean(lws323)
lw453 = mean(lws453)

err21 = stdev(lws21)
err33 = stdev(lws33)
err53 = stdev(lws53)
err83 = stdev(lws83)
err251 = stdev(lws251)
err323 = stdev(lws323)
err453 = stdev(lws453)

lws_0326 = np.array([lw21, lw33, lw53, lw83, lw251, lw323])*1e-12
errs_0326 = np.array([err21, err33, err53, err83, err251, err323])*1e-12



### 20250314 ###

lws100 = np.array([29.896, 15.429])*1e-12
lws150 = np.array([22.217, 13.674, 42.052, 31.426])*1e-12

lw100 = np.mean(lws100)
lw150 = np.mean(lws150)

err100 = stdev(lws100)
err150 = stdev(lws150)

lws_0314 = np.array([lw100, lw150])
errs_0314 = np.array([err100, err150])
################

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

the_good_data_points = np.concatenate([lws_0211, lws_0220[1:], lws_0218[:2], lws_0207[3:], [lws_0207[0]], np.array([23.575, 82.505])*1e-12])
good_errs = np.concatenate([err_0211, err_0220[1:], err_0218[:2], err_0207[3:], [err_0207[0]], np.array([1.927, 30.503])*1e-12])
the_good_lengths = np.concatenate([ls_0211, ls_0220[1:], ls_0218[:2], ls_0207[3:], [ls_0207[0]], np.array([718.787, 24.612])*1e-6])
#print(len(the_good_data_points))
#print(len(good_errs))
#print(len(the_good_lengths))

#### 20250326 ####

dlws_0326, slws_0326, bblws_0326 = calc_lws(l, params1_0326, params2_0326)
dlws_0326_p, slws_0326_p, bblws_0326_p = calc_lws(l, params1_0326+params1_0326_errs, params2_0326+params2_0326_errs)
dlws_0326_m, slws_0326_m, bblws_0326_m = calc_lws(l, params1_0326-params1_0326_errs, params2_0326-params2_0326_errs)

##################

dlws_0702, slws_0702, bblws_0702 = calc_lws(l, p1_0702, p2_0702)
dlws_1102, slws_1102, bblws_1102 = calc_lws(l, p1_1102, p2_1102)
dlws_1802, slws_1802, bblws_1802 = calc_lws(l, p1_1802, p2_1802)
dlws_2002, slws_2002, bblws_2002 = calc_lws(l, p1_2002, p2_2002)
dlws_0226, slws_0226, bblws_0226 = calc_lws(l, params1_0226, params2_0226)
dlws_0305, slws_0305, bblws_0305 = calc_lws(l, params1_0305, params2_0305)
dlws_origin, slws_origin, bblws_origin = calc_lws(l, params1_origin, params2_origin)

dlws_0702_p, slws_0702_p, bblws_0702_p = calc_lws(l, p1_0702+p1_errs, p2_0702+p2_errs)
dlws_1102_p, slws_1102_p, bblws_1102_p = calc_lws(l, p1_1102+p1_errs, p2_1102+p2_errs)
dlws_1802_p, slws_1802_p, bblws_1802_p = calc_lws(l, p1_1802+p1_errs, p2_1802+p2_errs)
dlws_2002_p, slws_2002_p, bblws_2002_p = calc_lws(l, p1_2002+p1_errs, p2_2002+p2_errs)
dlws_0226_p, slws_0226_p, bblws_0226_p = calc_lws(l, params1_0226+p1_errs, params2_0226+p2_errs)
dlws_0305_p, slws_0305_p, bblws_0305_p = calc_lws(l, params1_0305+p1_errs, params2_0305+p2_errs)
dlws_origin_p, slws_origin_p, bblws_origin_p = calc_lws(l, params1_origin+p1_errs, params2_origin+p2_errs)

dlws_0702_m, slws_0702_m, bblws_0702_m = calc_lws(l, p1_0702-p1_errs, p2_0702-p2_errs)
dlws_1102_m, slws_1102_m, bblws_1102_m = calc_lws(l, p1_1102-p1_errs, p2_1102-p2_errs)
dlws_1802_m, slws_1802_m, bblws_1802_m = calc_lws(l, p1_1802-p1_errs, p2_1802-p2_errs)
dlws_2002_m, slws_2002_m, bblws_2002_m = calc_lws(l, p1_2002-p1_errs, p2_2002-p2_errs)
dlws_0226_m, slws_0226_m, bblws_0226_m = calc_lws(l, params1_0226-p1_errs, params2_0226-p2_errs)
dlws_0305_m, slws_0305_m, bblws_0305_m = calc_lws(l, params1_0305-p1_errs, params2_0305-p2_errs)
dlws_origin_m, slws_origin_m, bblws_origin_m = calc_lws(l, params1_origin-p1_errs, params2_origin-p2_errs)

double_fano_lws = (dlws_0702+dlws_1102+dlws_1802+dlws_2002+dlws_0226+dlws_0305+dlws_origin+dlws_0326)/8
single_fano_lws = (slws_0702+slws_1102+slws_1802+slws_2002+slws_0226+slws_0305+slws_origin+slws_0326)/8
broadband_lws = (bblws_0702+bblws_1102+bblws_1802+bblws_2002+bblws_0226+bblws_0305+bblws_origin+bblws_0326)/8

double_fano_lws_p = (dlws_0702_p+dlws_1102_p+dlws_1802_p+dlws_2002_p+dlws_0226_p+dlws_0305_p+dlws_origin_p+dlws_0326_p)/8
single_fano_lws_p = (slws_0702_p+slws_1102_p+slws_1802_p+slws_2002_p+slws_0226_p+slws_0305_p+slws_origin_p+slws_0326_p)/8
broadband_lws_p = (bblws_0702_p+bblws_1102_p+bblws_1802_p+bblws_2002_p+bblws_0226_p+bblws_0305_p+bblws_origin_p+bblws_0326_p)/8

double_fano_lws_m = (dlws_0702_m+dlws_1102_m+dlws_1802_m+dlws_2002_m+dlws_0226_m+dlws_0305_m+dlws_origin_m+dlws_0326_m)/8
single_fano_lws_m = (slws_0702_m+slws_1102_m+slws_1802_m+slws_2002_m+slws_0226_m+slws_0305_m+slws_origin_m+slws_0326_m)/8
broadband_lws_m = (bblws_0702_m+bblws_1102_m+bblws_1802_m+bblws_2002_m+bblws_0226_m+bblws_0305_m+bblws_origin_m+bblws_0326_m)/8

lengths = np.linspace(10,1100,10000)*1e-6

params1_sim = [951, 951, 0.8, 0.5, 1e-6]

sim_lengths = np.array([10.459954599545997, 30.430634306343062, 90.34577345773458, 150.25909259092592, 300.04239042390424, 700.4123041230414, 1000.4525645256452])
sim_bblws = np.array([352.854, 121.299, 40.861, 24.568, 12.303, 5.271, 3.69])
sim_slws = np.array([55.746, 45.459, 25.924, 18.455, 10.690, 5.035, 3.605])
sim_dlws = np.array([30.987, 26.919, 19.224, 14.91, 9.517, 4.832, 3.529])
dlws, slws, bblws = calc_lws(lengths, params1_sim, params1_sim, losses=False)

losses = [0.0, 0.02, 0.06, 0.12, 0.24, 0.48] ## in percent
lws = [26.9, 32.126, 42.732, 59.052, 93.306, 169.794] ## in pm

linewidths = []

Ls = np.linspace(0, 0.5, 1000)

for L in Ls:
    rs = theoretical_reflection_values(params1_sim, λs, losses=True, loss_factor=L/2)[0]
    rs = [float(r) for r in rs]
    r_trans = np.max(rs)
    print(r_trans)
    rparams, _ = curve_fit(model, λs, rs, p0=params1_sim, maxfev=10000)
    rd = rparams[2]
    lw = double_fano_losses(30*1e-6, 951*1e-9, 2*(1-r_trans), 0.5, rd, 0.05, 0.05)*1e12
    linewidths.append(lw)



plt.figure(figsize=(10,6))
plt.scatter(losses, lws, color="forestgreen", marker=".", label="$\\delta \\lambda (L)$")
plt.plot(Ls, linewidths)
#plt.plot(lengths*1e6, bblws, linestyle="--", color="royalblue", alpha=0.5, label="broadband cavity")
#plt.plot(lengths*1e6, slws, linestyle="--", color="orangered", alpha=0.5, label="single fano cavity")
#plt.plot(lengths*1e6, dlws, linestyle="--", color="forestgreen", alpha=0.5, label="double fano cavity")
#plt.scatter(sim_lengths, sim_bblws, marker=".", color="royalblue", label="broadband cavity")
#plt.scatter(sim_lengths, sim_slws, marker=".", color="orangered", label="single fano cavity")
#plt.scatter(sim_lengths, sim_dlws, marker=".", color="forestgreen", label="double fano cavity")
#plt.errorbar(np.array(the_good_lengths)*1e6, np.array(the_good_data_points)*1e12, np.array(good_errs)*1e12, fmt=".", capsize=3, color="cornflowerblue", label="HWHM (measured)", zorder=7)
#plt.plot(l*1e6, broadband_lws, linestyle="--", color="royalblue", label="avg. broadband cavity")
#plt.plot(l*1e6,single_fano_lws, linestyle="--", label="avg. single fano cavity", color="orangered")
#plt.plot(l*1e6,double_fano_lws, linestyle="--", label= "avg. double fano cavity", color="forestgreen")
#plt.fill_between(l*1e6, broadband_lws_p, broadband_lws_m, color="royalblue", alpha=0.3)
#plt.fill_between(l*1e6, single_fano_lws_p, single_fano_lws_m, color="orangered", alpha=0.3)
#plt.fill_between(l*1e6, double_fano_lws_p, double_fano_lws_m, color="forestgreen", alpha=0.3)
#plt.errorbar(ls_0326*1e6, lws_0326*1e12, errs_0326*1e12, fmt=".", capsize=3, color="firebrick", label="HWHM (measured on 26/3)", zorder=7)
#plt.errorbar(ls_0314*1e6, lws_0314*1e12, errs_0314*1e12, fmt=".", capsize=3, color="firebrick", label="HWHM (measured on 14/3)", zorder=7)
#plt.plot(l*1e6, bblws_0326, linestyle="--", color="royalblue", label="broadband cavity")
#plt.plot(l*1e6, slws_0326, linestyle="--", label="single fano cavity", color="orangered")
#plt.plot(l*1e6, dlws_0326, linestyle="--", label= "double fano cavity", color="forestgreen")
#plt.fill_between(l*1e6, bblws_0326_p, bblws_0326_m, color="royalblue", alpha=0.3)
#plt.fill_between(l*1e6, slws_0326_p, slws_0326_m, color="orangered", alpha=0.3)
#plt.fill_between(l*1e6, dlws_0326_p, dlws_0326_m, color="forestgreen", alpha=0.3)
#plt.scatter(180,55)
#plt.scatter(251,48)
#plt.plot(l*1e6, dlws_0702)
#plt.plot(l*1e6, dlws_1102)
#plt.plot(l*1e6, dlws_1802)
#plt.plot(l*1e6, dlws_2002)
#plt.errorbar(ls_0207*1e6, lws_0207*1e12, err_0207*1e12, xerr=ls_0207_err*1e6, fmt=".", capsize=3, color="cornflowerblue", label="HWHM (measured on 7/2)")
#plt.errorbar(ls_0211*1e6, lws_0211*1e12, err_0211*1e12, xerr=ls_0211_err*1e6, fmt=".", capsize=3, color="orange", label="HWHM (measured on 11/2)")
#plt.errorbar(ls_0218*1e6, lws_0218*1e12, err_0218*1e12, xerr=ls_0218_err*1e6, fmt=".", capsize=3, color="limegreen", label="HWHM (measured on 18/2)")
#plt.errorbar(ls_0220*1e6, lws_0220*1e12, err_0220*1e12, xerr=ls_0220_err*1e6, fmt=".", capsize=3, color="magenta", label="HWHM (measured on 20/2)")
#plt.errorbar(ls_0225*1e6, lws_0225*1e12, err_0225*1e12, xerr=ls_0225_err*1e6, fmt=".", capsize=3, color="darkblue", label="HWHM (measured on 25/2)")
#plt.title("HWHM as a function of cavity length")
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



