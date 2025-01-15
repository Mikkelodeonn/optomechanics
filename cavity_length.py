import numpy as np


λ = 952.170

def L(λ,FSR):
    return λ**2/(2*FSR)*1e-3

print(L(λ,1.5), " μm")
