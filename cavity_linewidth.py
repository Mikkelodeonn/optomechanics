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

def lw_mirror(l: int, λres: float, r1: float, r2: float): 
    δγc = (λres**2/(8*np.pi*l)) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    return δγc

def lw_mirror_single(l: int, λres: float, Tm: float, Tg: float, L: float): 
    δγc = (λres**2/(8*np.pi*l)) * (Tg + Tm + L) 
    return δγc

def lw_fano(l: int, λres: float, γλ: float, rd: float, r1: float, r2: float): 
    δγc = ((λres**2)/(8*np.pi*l)) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    δγg = ((γλ/(2*(1-rd)))) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

def lw_single_fano(l: int, λres: float, γλ: float, rd: float, Tm: float, Tg: float, L: float): 
    δγc = ((λres**2)/(8*np.pi*l)) * (Tg + Tm + L) 
    δγg = ((γλ/(2*(1-rd)))) * (Tg + Tm + L) 
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ 

def double_fano(l: int, λres: float, γλ: float, rd: float, r1: float, r2: float):
    δγc = ((λres**2)/(8*np.pi*l)) * (1 - r1 + 1 - r2) #(Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (1 - r1 + 1 - r2)*0.5 #(Tg + Tm + L) * 0.5
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ

def double_fano_losses(l: int, λres: float, L: float, γλ: float, rd: float, t1: float, t2: float):
    δγc = ((λres**2)/(8*np.pi*l)) * (t1 + t2 + L) #(Tg + Tm + L) 
    δγg = (γλ/(2*(1-rd))) * (t1 + t2 + L)*0.5 #(Tg + Tm + L) * 0.5
    δγ = 1/((1/δγc) + (1/δγg)) 
    return δγ

def calc_lws(l, params1, params2, losses=True):
    λ0_1 = params1[0]; γ_1 = params1[3]
    λ0_2 = params2[0]; γ_2 = params2[3]

    λt = np.array([0.5*λ0_1 + 0.5*λ0_2]) 

    #t_M3_trans = model(λt, *params1)
    #t_M5_trans = model(λt, *params2)
    r_M3_trans = theoretical_reflection_values(params1, λt, losses=losses)[0][0]
    r_M5_trans = theoretical_reflection_values(params2, λt, losses=losses)[0][0]

    r1s = theoretical_reflection_values(params1, λs, losses=losses)[0]
    r2s = theoretical_reflection_values(params2, λs, losses=losses)[0]
    r1s = [float(r) for r in r1s]
    r2s = [float(r) for r in r2s]
    #print(np.max(r1s))
    #print(np.max(r2s))

    rparams1, _ = curve_fit(model, λs, r1s, p0=p0)
    rparams2, _ = curve_fit(model, λs, r2s, p0=p0)

    λres = 0.5*λ0_1*1e-9 + 0.5*λ0_2*1e-9
    γλ = (0.5*γ_1*1e-9 + 0.5*γ_2*1e-9)
    rd1 = rparams1[2]
    rd2 = rparams2[2]
    rd = (rd1 + rd2)/2

    double_fano_lws = double_fano(l, λres, γλ, rd, r_M3_trans, r_M5_trans)*1e12

    single_fano_lws = lw_fano(l, λres, γλ, rd, r_M3_trans, r_M5_trans)*1e12

    mirror_lws = lw_mirror(l, λres, r_M3_trans, r_M5_trans)*1e12

    return [double_fano_lws, single_fano_lws, mirror_lws]

def calc_lws_single(l, params, Tm: float, losses=True): ## define L and Tm and Tg
    λ0 = params[0]; γ = params[3]

    Tg = model(λ0, *params)
    rt = theoretical_reflection_values(params, [λ0], losses=losses)[0][0]

    rs = theoretical_reflection_values(params, λs, losses=losses)[0]
    rs = [float(r) for r in rs]

    L = 1 - rt - Tg

    rparams, _ = curve_fit(model, λs, rs, p0=p0)

    λres = λ0*1e-9
    γλ = γ*1e-9
    rd = rparams[2]

    single_fano_lws = lw_single_fano(l, λres, γλ, rd, Tm, Tg, L)*1e12

    mirror_lws = lw_mirror_single(l, λres, Tm, Tg, L)*1e12

    return [single_fano_lws, mirror_lws]

l = np.linspace(10,1000,1000)*1e-6

M3 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M3/400_M3 trans.txt")
M5 = fano("/Users/mikkelodeon/optomechanics/400um gratings/Data/M5/400_M5 trans.txt")

M5_single = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/grating trans/M5_trans.txt")
M5_single_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/grating trans/M5_trans_PI.txt")
M5_single_norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/normalization/grating_trans_PI.txt")
M5_single_norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Single fano cavity/Data/20250512/normalization/grating_trans_PI.txt")
M5_single[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5_single[:,1], M5_single_PI[:,1], M5_single_norm[:,1], M5_single_norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

M5_single_params, pcov_M5_single = curve_fit(model, M5_single[:,0], M5_single[:,1], p0=[951.8, 951.8, 0.1, 0.5, 1e-6])
M5_single_errs = np.sqrt(np.diag(pcov_M5_single))

params1_origin, params1_origin_errs = M3.lossy_fit([952,952,0.6,1,0.1], with_errors=True)
params2_origin, params2_origin_errs = M5.lossy_fit([952,952,0.6,1,0.1], with_errors=True)



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

lengths = np.linspace(5,900,10000)*1e-6 ## 10, 30, 90, 270, 810, 2430

params1_sim = [951.216982, 951.355926, 0.817946804, 0.527290175, 1.02607530e-06]

sim_lengths = np.array([10.01, 30.45, 90.37, 270.15, 810.44])
sim_bblws = np.array([352.97, 121.333, 40.87, 13.671, 4.557])
sim_slws = np.array([50.731, 39.951, 24.391, 11.107, 4.205])
sim_dlws = np.array([25.94, 22.89, 16.845, 9.244, 3.886])
dlws, slws, bblws = calc_lws(lengths, params1_sim, params1_sim, losses=False)

losses = np.array([0.98, 1.98, 3.94, 7.88, 15.66]) ## in percent
lws = np.array([26.607, 29.096, 34.098, 44.184, 64.725]) ## in pm

linewidths = []

Ls = np.linspace(0.8, 18, 1000)

#for L in Ls:
#    L = L*1e-2
#    λres = params1_sim[0]*1e-9 ## nm -> m
#    γλ = params1_sim[3]*1e-9 ## nm -> m
#    length = 30*1e-6 ## um -> m
#    td = params1_sim[2]#

#    rs = theoretical_reflection_values(params1_sim, λs, losses=True, loss_factor=L/2)[0]
#    rs = [float(r) for r in rs]

#    r_trans = theoretical_reflection_values(params1_sim, np.array([params1_sim[0]]), losses=True, loss_factor=L/2)[0][0]
#    t_trans = model(np.array([params1_sim[0]]), *params1_sim)

    #print(r_trans, t_trans)

#    rparams, _ = curve_fit(model, λs, rs, p0=params1_sim, maxfev=10000)
#    rd = np.sqrt(1-td**2)#rparams[2]

    #print(r_trans**2 + t_trans**2 + L)
    #cavity_losses = L #2*(1-r_trans-t_trans)
    #print(cavity_losses)
#    lw = double_fano_losses(length, λres, L, γλ, rd, t_trans, t_trans)*1e12
    #lw = double_fano(length, λres, γλ, rd, r_trans, r_trans)*1e12
#    linewidths.append(lw)

lws_0422_20 = np.array([79.413, 106.497, 92.136, 85.447, 98.366, 96.821, 109.023])
lws_0422_40 = np.array([60.243, 116.723])
lws_0422_75 = np.array([85.333, 52.313, 59.151])
lws_0422_100 = np.array([105.658, 89.409, 57.386])
lws_0422_110 = np.array([109.218, 118.252])
lws_0422_225 = np.array([71.717, 41.860, 64.719])
lws_0422_500 = np.array([64.260, 64.377, 57.153])

lw_0422_20 = np.mean(lws_0422_20)
lw_0422_40 = np.mean(lws_0422_40)
lw_0422_75 = np.mean(lws_0422_75)
lw_0422_100 = np.mean(lws_0422_100)
lw_0422_110 = np.mean(lws_0422_110)
lw_0422_225 = np.mean(lws_0422_225)
lw_0422_500 = np.mean(lws_0422_500)

err_0422_20 = stdev(lws_0422_20)
err_0422_40 = stdev(lws_0422_40)
err_0422_75 = stdev(lws_0422_75)
err_0422_100 = stdev(lws_0422_100)
err_0422_110 = stdev(lws_0422_110)
err_0422_225 = stdev(lws_0422_225)
err_0422_500 = stdev(lws_0422_500)

lws_0422 = np.array([lw_0422_20, lw_0422_40, lw_0422_75, lw_0422_100, lw_0422_110, lw_0422_225, lw_0422_500])
errs_0422 = np.array([err_0422_20, err_0422_40, err_0422_75, err_0422_100, err_0422_110, err_0422_225, err_0422_500])
ls_0422 = np.array([20,40,75,100,110,225,500]) ## approximate only!!

### Single fano measurements 20250512

single_lengths = np.array([24.05, 57.40, 116.31, 211.98, 385.96])
lengths_err = np.array([0.54, 1.55, 1.19, 3.16, 2.90])
sim_lws_single = np.array([34.603, 25.785, 17.583, 11.563, 7.127])

lws_5um = np.array([28.766, 32.682, 38.827, 39.827])
lws_60um = np.array([33.907, 18.958, 22.349, 20.412])
lws_120um = np.array([18.219, 15.942, 20.417])
lws_220um = np.array([13.372, 17.274, 17.845, 11.982])
lws_380um = np.array([7.843, 9.433, 10.238, 9.796])

lws_5um_err = stdev(lws_5um)
lws_60um_err = stdev(lws_60um)
lws_120um_err = stdev(lws_120um)
lws_220um_err = stdev(lws_220um)
lws_380um_err = stdev(lws_380um)

lw_5um = np.mean(lws_5um)
lw_60um = np.mean(lws_60um)
lw_120um = np.mean(lws_120um)
lw_220um = np.mean(lws_220um)
lw_380um = np.mean(lws_380um)

lws_0512 = [lw_5um, lw_60um, lw_120um, lw_220um, lw_380um]
errs_0512 = [lws_5um_err, lws_60um_err, lws_120um_err, lws_220um_err, lws_380um_err]

slws_0512, bblws_0512 = calc_lws_single(l, params2_origin, Tm = 0.005)
slws_0512_m, bblws_0512_m = calc_lws_single(l, params2_origin-params2_origin_errs, Tm=0.005)
slws_0512_p, bblws_0512_p = calc_lws_single(l, params2_origin+params2_origin_errs, Tm=0.005)

############################## 20250326 ##############################  

ls_0326 = np.array([21.45, 32.71, 52.22, 80.63, 244.89, 320.25, 450.48])*1e-6

lws21 = np.array([68.829, 74.187, 48.436, 48.777, 71.753, 65.673, 46.808, 54.260, 61.337, 68.750, 56.628, 63.813, 56.939, 62.304])
lws33 = np.array([37.528, 78.588, 52.193, 58.226, 74.592, 68.143, 47.219, 54.553, 62.793, 62.023, 45.564, 71.124, 60.410, 85.821, 59.870])
lws53 = np.array([55.869, 54.361, 76.880, 51.404, 34.751, 52.525, 35.218, 48.779, 69.359, 50.462]) 
lws83 = np.array([54.756, 53.092, 78.517, 62.764, 58.174, 58.751, 47.061, 52.086, 62.370, 47.063, 82.582]) 
lws251 = np.array([58.619, 77.617, 76.209, 67.744, 51.382, 45.476, 77.490, 80.531, 76.911, 77.461])
lws323 = np.array([52.294, 48.373, 77.932, 65.202, 54.043, 53.608])
lws453 = np.array([78.216, 75.434, 79.108, 69.476, 61.065, 49.889, 45.614, 48.903])

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

lws_0326 = np.array([lw21, lw33, lw53, lw83, lw251, lw323, lw453])*1e-12
errs_0326 = np.array([err21, err33, err53, err83, err251, err323, err453])*1e-12

############################## 20250523 ##############################    

M3_0523 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/grating trans/M3_trans.txt")
M5_0523 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/grating trans/M5_trans.txt")
M3_0523_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/grating trans/M3_trans_PI.txt")
M5_0523_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/grating trans/M5_trans_PI.txt")
norm_0523 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/normalization/grating_trans.txt")
norm_0523_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250523/normalization/grating_trans_PI.txt")

M3_0523[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M3_0523[:,1], M3_0523_PI[:,1], norm_0523[:,1], norm_0523_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 
M5_0523[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(M5_0523[:,1], M5_0523_PI[:,1], norm_0523[:,1], norm_0523_PI[:,1])] ## norm. with respect to trans. w/o a cavity.

params1_0523, pcov1_0523 = curve_fit(model, M3_0523[:,0], M3_0523[:,1], p0=p0)
params2_0523, pcov2_0523 = curve_fit(model, M5_0523[:,0], M5_0523[:,1], p0=p0)

params1_0523_errs = np.sqrt(np.diag(pcov1_0523))
params2_0523_errs = np.sqrt(np.diag(pcov2_0523))

lengths_0523 = np.array([17.04, 65.75, 120.64, 239.63, 308.03, 539.10, 976.44])
lengths_err_0523 = np.array([0.23, 0.37, 0.73, 1.04, 0.92, 2.33, 1.88])
sim_lws_0523 = np.array([71.791, 53.384, 41.106, 27.342, 22.921, 14.763, 8.791])

lws_30um_0523 = np.array([63.891, 80.831, 86.779, 73.419])
lws_70um_0523 = np.array([83.233, 70.400, 77.562, 82.530, 88.070]) 
lws_120um_0523 = np.array([75.893, 75.341, 77.813, 71.019, 48.475, 69.561, 79.089]) 
lws_250um_0523 = np.array([74.371, 61.325, 51.241, 67.691, 54.200, 60.248]) 
lws_320um_0523 = np.array([57.924, 51.095, 65.618, 54.529, 57.702, 56.557]) 
lws_550um_0523 = np.array([30.623, 36.089, 36.920, 34.302, 36.522])
lws_1000um_0523 = np.array([34.471, 25.998, 29.564, 26.449, 27.875]) 

lws_30um_err_0523 = stdev(lws_30um_0523)
lws_70um_err_0523 = stdev(lws_70um_0523)
lws_120um_err_0523 = stdev(lws_120um_0523)
lws_250um_err_0523 = stdev(lws_250um_0523)
lws_320um_err_0523 = stdev(lws_320um_0523)
lws_550um_err_0523 = stdev(lws_550um_0523)
lws_1000um_err_0523 = stdev(lws_1000um_0523)

lw_30um_0523 = np.mean(lws_30um_0523)
lw_70um_0523 = np.mean(lws_70um_0523)
lw_120um_0523 = np.mean(lws_120um_0523)
lw_250um_0523 = np.mean(lws_250um_0523)
lw_320um_0523 = np.mean(lws_320um_0523)
lw_550um_0523 = np.mean(lws_550um_0523)
lw_1000um_0523 = np.mean(lws_1000um_0523)

lws_0523 = [lw_30um_0523, lw_70um_0523, lw_120um_0523, lw_250um_0523, lw_320um_0523, lw_550um_0523, lw_1000um_0523]
errs_0523 = [lws_30um_err_0523, lws_70um_err_0523, lws_120um_err_0523, lws_250um_err_0523, lws_320um_err_0523, lws_550um_err_0523, lws_1000um_err_0523]

print(np.array(lws_0523)-sim_lws_0523)

dlws_0523, slws_0523, bblws_0523 = calc_lws(l, params1_0523, params2_0523)
dlws_0523_p, slws_0523_p, bblws_0523_p = calc_lws(l, params1_0523+params1_0523_errs, params2_0523+params2_0523_errs)
dlws_0523_m, slws_0523_m, bblws_0523_m = calc_lws(l, params1_0523-params1_0523_errs, params2_0523-params2_0523_errs)

#####################################################################   
############################ Averaged ###############################

#bblws = (bblws_0523 + bblws_0326)/2
#slws = (slws_0523 + slws_0326)/2
#dlws = (dlws_0523 + dlws_0326)/2

#bblws_p = (bblws_0523_p + bblws_0326_p)/2
#slws_p = (slws_0523_p + slws_0326_p)/2
#dlws_p = (dlws_0523_p + dlws_0326_p)/2

#bblws_m = (bblws_0523_m + bblws_0326_m)/2
#slws_m = (slws_0523_m + slws_0326_m)/2
#dlws_m = (dlws_0523_m + dlws_0326_m)/2

##################################################################### 

plt.figure(figsize=(10,7))
#plt.errorbar(ls_0326*1e6, lws_0326*1e12, errs_0326*1e12, fmt=".", capsize=3, color="darkviolet", label="data", zorder=7)
#plt.errorbar(lengths_0523, lws_0523, errs_0523, fmt=".", capsize=3, color="firebrick", label="data", zorder=7) # measured on 23/5
#plt.scatter(lengths_0523, sim_lws_0523, marker=".", color="black", label="simulation")
#plt.plot(l*1e6, bblws_0523, linestyle="--", color="royalblue", label="broadband")
#plt.plot(l*1e6, slws_0523, linestyle="--", label="single Fano", color="orangered")
#plt.plot(l*1e6, dlws_0523, linestyle="--", label= "double Fano", color="forestgreen")
#plt.fill_between(l*1e6, bblws_0523_p, bblws_0523_m, color="royalblue", alpha=0.1)
#plt.fill_between(l*1e6, slws_0523_p, slws_0523_m, color="orangered", alpha=0.1)
#plt.fill_between(l*1e6, dlws_0523_p, dlws_0523_m, color="forestgreen", alpha=0.1)

#plt.plot(l*1e6, bblws_0326, linestyle="--", color="royalblue", label="broadband")
#plt.plot(l*1e6, slws_0326, linestyle="--", label="single Fano", color="orangered")
#plt.plot(l*1e6, dlws_0326, linestyle="--", label= "double Fano", color="forestgreen")
#plt.fill_between(l*1e6, bblws_0326_p, bblws_0326_m, color="royalblue", alpha=0.1)
#plt.fill_between(l*1e6, slws_0326_p, slws_0326_m, color="orangered", alpha=0.1)
#plt.fill_between(l*1e6, dlws_0326_p, dlws_0326_m, color="forestgreen", alpha=0.1)

#plt.plot(l*1e6, bblws_origin, linestyle="--", color="royalblue", label="broadband")
#plt.plot(l*1e6, slws_origin, linestyle="--", label="single Fano", color="orangered")
#plt.plot(l*1e6, dlws_origin, linestyle="--", label= "double Fano", color="forestgreen")
#plt.fill_between(l*1e6, bblws_origin_p, bblws_origin_m, color="royalblue", alpha=0.1)
#plt.fill_between(l*1e6, slws_origin_p, slws_origin_m, color="orangered", alpha=0.1)
#plt.fill_between(l*1e6, dlws_origin_p, dlws_origin_m, color="forestgreen", alpha=0.1)


plt.errorbar(single_lengths, lws_0512, errs_0512, lengths_err, fmt=".", capsize=3, label="data")
plt.plot(l*1e6, slws_0512, linestyle="--", color="orangered", label="single fano")
plt.plot(l*1e6, bblws_0512, linestyle="--", color="royalblue", label="broadband")
plt.fill_between(l*1e6, bblws_0512_p, bblws_0512_m, color="royalblue", alpha=0.1)
plt.fill_between(l*1e6, slws_0512_p, slws_0512_m, color="orangered", alpha=0.1)
plt.scatter(single_lengths, sim_lws_single, marker="o", color="black", label="simulation")


#plt.scatter(losses, lws, color="forestgreen", marker=".", label="simulated")
#plt.plot(Ls, linewidths, color="forestgreen", alpha=0.5, label="analytical")


#plt.plot(lengths*1e6, bblws, linestyle="--", color="royalblue", alpha=0.5, label="broadband analytical")
#plt.plot(lengths*1e6, slws, linestyle="--", color="orangered", alpha=0.5, label="single Fano analytical")
#plt.plot(lengths*1e6, dlws, linestyle="--", color="forestgreen", alpha=0.5, label="double Fano analytical")
#plt.scatter(sim_lengths, sim_bblws, marker=".", color="royalblue", label="broadband sim.")
#plt.scatter(sim_lengths, sim_slws, marker=".", color="orangered", label="single Fano sim.")
#plt.scatter(sim_lengths, sim_dlws, marker=".", color="forestgreen", label="double Fano sim.")
#plt.errorbar(np.array(the_good_lengths)*1e6, np.array(the_good_data_points)*1e12, np.array(good_errs)*1e12, fmt=".", capsize=3, color="cornflowerblue", label="HWHM (measured)", zorder=7)
#plt.plot(l*1e6, broadband_lws, linestyle="--", color="royalblue", label="avg. broadband cavity")
#plt.plot(l*1e6,single_fano_lws, linestyle="--", label="avg. single fano cavity", color="orangered")
#plt.plot(l*1e6,double_fano_lws, linestyle="--", label= "avg. double fano cavity", color="forestgreen")
#plt.fill_between(l*1e6, broadband_lws_p, broadband_lws_m, color="royalblue", alpha=0.3)
#plt.fill_between(l*1e6, single_fano_lws_p, single_fano_lws_m, color="orangered", alpha=0.3)
#plt.fill_between(l*1e6, double_fano_lws_p, double_fano_lws_m, color="forestgreen", alpha=0.3)
#plt.errorbar(ls_0422, lws_0422, errs_0422, fmt=".", capsize=3, color ="magenta", label="HWHM (measured on 22/4)")
#plt.errorbar(ls_0314*1e6, lws_0314*1e12, errs_0314*1e12, fmt=".", capsize=3, color="firebrick", label="HWHM (measured on 14/3)", zorder=7)
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
plt.xlabel("cavity length [μm]", fontsize=28)
#plt.xlabel("L [%]", fontsize=28)
plt.ylabel("HWHM [pm]", fontsize=28)
plt.xscale("log")
plt.yscale("log")
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
#plt.ticklabel_format(style='plain', axis="both")
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
plt.subplots_adjust(bottom=0.3, left=0.15)
plt.grid(True, which="both", ls="--", alpha=0.2)
plt.show()



