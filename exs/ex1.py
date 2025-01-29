import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# def le sumbol
x = sp.symbols('x')

# la fonction
f = 2 * (x + 2) * sp.sin(3 * x + 1)

# la dirivée
df = sp.diff(f, x)

_fun_f = sp.lambdify(x, f, "numpy")
_fun_df = sp.lambdify(x, df, "numpy")



# Descente de gradient
def des_grad(x0,alpha,fun_df,max):
    x_current = x0
    points = [x_current]  
    for i in range(max):
        grad_x = fun_df(x_current)
        x_new = x_current - alpha * grad_x  
        points.append(x_new)
        x_current = x_new
    return points

def tracer(points,fun_f):
    X = np.linspace(-4, 4, 400)
    Y = fun_f(X)
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, label="f(x)")
    # plt.scatter(points, fun_f(np.array(points)), color="red", marker="o", label="Points de la descente de gradient")
    plt.scatter(points, [fun_f(i) for i in points], color="red", marker="o", label="Points de la descente de gradient")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Descente de Gradient pour f(x) = 2(x+2)sin(3x+1)")
    plt.show()

# parametres
print("Descente de Gradient pour f(x) = 2(x+2)sin(3x+1)")
alpha = float(input('Entrer le taux d\'apprentissage alpha :')) 
max_iter = int(input('Entrer le nombre maximal d\'itérations max:')) 
x0 = 0
if (alpha>0 and max_iter>=100):
    tracer(des_grad(x0,alpha,_fun_df,max_iter),_fun_f)
else:
    print('alpha doit etre positif ! et le max doit etre >=100 !!!!')

