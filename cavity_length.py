import numpy as np


λ = 955.576

def L(λ,FSR):
    return λ**2/(2*FSR)*1e-3

print(L(λ,7), " μm")