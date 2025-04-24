import numpy as np

cavity_lengths = [21, 33, 53, 83, 251, 323, 453]
number_of_scans = [14, 15, 10, 11, 10, 6, 8]

day_of_measurement = 20250326

for length, number in zip(cavity_lengths, number_of_scans): 
    for i in range(number):
        data = data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/"+str(length)+"um/s"+str(i+1)+".txt")
        PI_data = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/"+str(length)+"um/s"+str(i+1)+"_PI.txt")
        norm = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalization/short_scan.txt")
        norm_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalization/short_scan.txt")

        if not np.allclose(data[:,0], norm[:,0]):
            raise Exception("Normalization and data files do not match!")
        
        data[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(data[:,1], PI_data[:,1], norm[:,1], norm_PI[:,1])] ## norm. with respect to trans. w/o a cavity. 

        output = np.column_stack((data[:,0], data[:,1]))

        np.savetxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalized data/"+str(length)+"um/s" + str(i+1) + "_normalized.txt", output, fmt="%.6e", comments='')

g1 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/grating trans. spectra/M3_trans.txt")
g1_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/grating trans. spectra/M3_trans_PI.txt")
g2 = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/grating trans. spectra/M5_trans.txt")
g2_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/grating trans. spectra/M5_trans_PI.txt")
norm_g = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalization/grating_trans.txt")
norm_g_PI = np.loadtxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalization/grating_trans_PI.txt")

g1[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(g1[:,1], g1_PI[:,1], norm_g[:,1], norm_g_PI[:,1])]  
g2[:,1] = [(d/pi)/(n/pi_) for d,pi,n,pi_ in zip(g2[:,1], g2_PI[:,1], norm_g[:,1], norm_g_PI[:,1])] 

g1_output = np.column_stack((g1[:,0], g1[:,1]))
g2_output = np.column_stack((g2[:,0], g2[:,1]))

np.savetxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalized data/grating trans. spectra/M3_trans_normalized.txt", g1_output, fmt="%.6e", comments='')

np.savetxt("/Users/mikkelodeon/optomechanics/Double fano cavity/M3+M5/data/"+str(day_of_measurement)+"/normalized data/grating trans. spectra/M5_trans_normalized.txt", g2_output, fmt="%.6e", comments='')