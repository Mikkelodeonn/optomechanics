import numpy as np


λ = 951.625 #nm

def L(λ,FSR):
    return λ**2/(2*FSR)*1e-3

print(L(λ,4.2), " μm")
