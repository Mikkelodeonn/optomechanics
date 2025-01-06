import numpy as np


λ = 955.000

def L(λ,FSR):
    return λ**2/(2*FSR)*1e-3

print(L(λ,0.8), " μm")
