import sympy as sym
import numpy as np
k= input('donner l\'ordre de la dérivation de la fonction x^2ln(1+x): ')
x = sym.Symbol('x')
g= np.pow(x,2)*sym.ln(1+x)
b=sym.diff(g, x, k)

print(b)
