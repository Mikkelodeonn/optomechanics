import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/21um/s3.txt")#[left:right]
PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/21um/s3_PI.txt")#[left:right]
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/short_scan.txt")#[left:right]
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/20250326/normalization/short_scan.txt")#[left:right]

if not np.allclose(data[:,0], norm[:,0]):
    raise Exception("Normalization and data files do not match!") 

#data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

plt.figure(figsize=(10,7))

#plt.plot(data[:,0], data[:,1], color="cornflowerblue", label="transmission, $P_T$")
#plt.plot(PI_data[:,0], PI_data[:,1], color="maroon", label="incident light, $P_I$")
#plt.plot(norm[:,0], norm[:,1], color="orangered", label="transmission, $P_T^{\prime}$")
plt.plot(norm_PI[:,0], norm_PI[:,1], color="firebrick", label="incident light, $P_I^{\prime}$")
plt.xlabel("wavelength [nm]", fontsize=28)
plt.ylabel("signal [V]", fontsize=28)
#plt.ylabel("norm. trans.", fontsize=28)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.3, left=0.15)
plt.show()
