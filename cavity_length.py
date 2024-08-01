import numpy as np


λ = 951.930

def L(λ,FSR):
    return λ**2/(2*FSR)*1e-3

print(L(λ,20.5), " μm")