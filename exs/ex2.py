import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Définition des variables symboliques
x, y = sp.symbols('x y')

# Déf de la fonction f(x, y)
f = x**2 + y**2 + 3*x*y + 2*x + 4*y

# Calcul des dérivées partielles
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

# Conversion en fonctions Python pour l'évaluation rapide
fun_df_dx = sp.lambdify((x, y), df_dx, "numpy")
fun_df_dy = sp.lambdify((x, y), df_dy, "numpy")
fun_f = sp.lambdify((x, y), f, "numpy")



# Descente de gradient
def des_grad(x0,y0,alpha,fun_df_dx,fun_df_dy,max):
    x_current, y_current = x0, y0
    points = [(x_current, y_current)] 

    for i in range(max):
        grad_x = fun_df_dx(x_current, y_current)
        grad_y = fun_df_dy(x_current, y_current)
        x_new = x_current - alpha * grad_x
        y_new = y_current - alpha * grad_y
        points.append((x_new, y_new))
        x_current, y_current = x_new, y_new
    return points


def tracer (points,fun_f):
    X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    Z = fun_f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, cmap="viridis")
    px, py = zip(*points)
    plt.plot(px, py, marker="o", color="red", label="Descente de gradient")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Descente de gradient pour f(x, y)")
    plt.show()

# parametres
print("Descente de Gradient pour f(x,y)")
alpha = float(input('Entrer le taux d\'apprentissage alpha :')) 
max_iter = int(input('Entrer le nombre maximal d\'itérations max:')) 
x0 , y0= 0,0
if (alpha>0 and max_iter>=20):
    tracer(des_grad(x0,y0,alpha,fun_df_dx,fun_df_dy,max_iter),fun_f)
else:
    print('alpha doit etre positif ! et le max doit etre >=20 !!!!')
