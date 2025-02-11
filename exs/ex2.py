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
fun_df_dx = sp.lambdify((x, y), df_dx)
fun_df_dy = sp.lambdify((x, y), df_dy )
fun_f = sp.lambdify((x, y), f)



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
    X, Y = np.meshgrid(np.linspace(-20, 20, 1000), np.linspace(-20, 20, 1000))
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

def tracer_3d(points, fun_f):
    X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
    Z = fun_f(X, Y)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    px, py = zip(*points)
    pz = [fun_f(x, y) for x, y in points]
    ax.scatter(px, py, pz, color='red', marker='o', s=50, label='Descente de gradient')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Descente de gradient pour f(x, y)')
    ax.legend()
    
    plt.show()

# parametres
print("Descente de Gradient pour f(x,y)")
alpha = float(input('Entrer le taux d\'apprentissage alpha :')) 
max_iter = int(input('Entrer le nombre maximal d\'itérations max:')) 
x0 , y0= 0,0
if (alpha>0 and max_iter>=20):
    tracer_3d(des_grad(x0,y0,alpha,fun_df_dx,fun_df_dy,max_iter),fun_f)
else:
    print('alpha doit etre positif ! et le max doit etre >=20 !!!!')
