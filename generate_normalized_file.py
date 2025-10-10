import numpy as np
import matplotlib.pyplot as plt

day_of_measurement = 20250207

data = data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/238um/238l.txt")
PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/238um/238l_PI.txt")
norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalization/long_scan.txt")
norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalization/long_scan_PI.txt")

if not np.allclose(data[:,0], norm[:,0]):
    raise Exception("Normalization and data files do not match!")
        
data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

output = np.column_stack((data[:,0], data[:,1]))

np.savetxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/normalized_data_for_paper/238um_long_scan_normalized.txt", output, fmt="%.6e", comments='')

fig, ax = plt.subplots()

ax.scatter(data[:,0], data[:,1], marker=".")
plt.show()
