import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x,y,z =sp.symbols("x y z")
f= (x**2)+(y**2)+(z**2)+(x*y)+(y*z)+(z*x)+x+y+z
df_dx=sp.diff(f,x)
df_dy=sp.diff(f,y)
df_dz=sp.diff(f,z)
fun_df_dx=sp.lambdify((x,y,z),df_dx,"numpy")
fun_df_dy=sp.lambdify((x,y,z),df_dy,"numpy")
fun_df_dz=sp.lambdify((x,y,z),df_dz,"numpy")
fun_f=sp.lambdify((x,y,z),f,"numpy")

def des_grad(fun_df_dx,fun_df_dy,fun_df_dz,alpha,max):
    cur_x,cur_y,cur_z=0,0,0
    points=[(cur_x,cur_y,cur_z)]
    for i in range(max):
        grad_x,grad_y,grad_z=fun_df_dx(cur_x,cur_y,cur_z),fun_df_dy(cur_x,cur_y,cur_z),fun_df_dz(cur_x,cur_y,cur_z)
        
        new_x=cur_x-alpha*grad_x
        new_y=cur_y-alpha*grad_y
        new_z=cur_z-alpha*grad_z
        
        points.append((new_x,new_y,new_z))
        
        cur_x,cur_y,cur_z=new_x,new_y,new_z
        
    return points

# parametres
print("Descente de Gradient pour f(x,y,z)")
alpha = float(input('Entrer le taux d\'apprentissage alpha :')) 
max_iter = int(input('Entrer le nombre maximal d\'itérations max:')) 
if (alpha>0 and max_iter>=20):
    print(des_grad(fun_df_dx,fun_df_dy,fun_df_dz,alpha,max_iter))
else:
    print('alpha doit etre positif ! et le max doit etre >=20 !!!!')

