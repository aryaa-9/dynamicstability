import random
import sympy as sp
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def energyfunc(S):
    x, y, c, k, a, m, n = S
    
    #Function 1 (Energy function):
    #ans = (1/2)*k*x**2+(1/2)*y**2
    
    #Function 2: 
    #ans = x**2+(3*x*y)+(3*y**2)
    
    #Function 3:
    ans = x**(2*m)+a*y**(2*n)
    
    return ans
    
def Sdot(t, S):
    x, y, c, k, a, m, n = S
    
    #Function 1 (Damped Harmonic oscillator): 
    #return [y, -k*x-c*(y**3)*(1+(x**2))]
    
    #Function 2: 
    #return [-2*x-3*y+x**2, x+y]
    
    #Function 3: 
    return [-x+2*y**3-2*y**4, -x-y+x*y]

def ddtenergy(S):
    x, y, c, k, a, m, n, t = S
    
    dfdx = sp.diff(energyfunc((x, y, c, k, a, m, n)), x)
    dfdy = sp.diff(energyfunc((x, y, c, k, a, m, n)), y)
    
    xdot=Sdot(t, (x, y, c, k, a, m, n))[0]
    ydot=Sdot(t, (x, y, c, k, a, m, n))[1]
    
    
    dfdt = dfdx * xdot + dfdy * ydot
    
    return dfdt
    

def equipts(dSdt, S):
    x, y, c, k, a, m, n = S
    dxdt, dydt=dSdt
    eq1=sp.Eq(dxdt, 0)
    eq2=sp.Eq(dydt, 0)
    
    # Solving for x and y
    sol=sp.solve((eq1, eq2), (x, y))
    return sol

def evalV(V, S, sols, l):
    results = []
    x, y, c, k, a, m, n = S
    for i in range(l):
        V00=V.subs({x: sols[i][0], y: sols[i][1]})
        results.append(V00)
    return results

def plotting(X, Y, Z, zlab, title1, title2, sol, i):
    if (i==-1):
        #Surface PLot
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(zlab)
        ax.set_title(title1)
        plt.show()
        
        #Contour Plot
        plt.figure(figsize=(10,7))
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title2)
        plt.show()
    else:
        xval=round(sol[i][0],2)
        yval=round(sol[i][1],2)
        tup=(xval, yval)
        #Surface PLot
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(zlab)
        ax.set_title(title1+" near "+str((xval, yval)))
        plt.show()
        
        #Contour Plot
        plt.figure(figsize=(10,7))
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title2+" near "+str((xval, yval)))
        plt.show()
   
    
def main():
    x, y, k, c, t, a, m, n = sp.symbols('x y k c t a m n')
    
    #V(x,y)
    V = energyfunc((x, y, c, k, a, m, n))
    #sp.pretty_print(V)
    
    dSdt = Sdot(t, (x, y, c, k, a, m, n))
    #sp.pretty_print(dSdt)

    sol1 = equipts(dSdt, (x, y, c, k, a, m, n))
    l = len(sol1)
    
    
    sol=[]
    for i in range(l):
        x0, y0 = sol1[i]
        x0_real = float(sp.re(x0))
        y0_real = float(sp.re(y0))
        
        x01=round(x0_real,10)
        y01=round(y0_real,10)
        sol.append((x01,y01))
    
    sol=list(set(sol))
    l = len(sol)
    
    results=evalV(V, (x, y, c, k, a, m, n), sol, l)
    funcs=[]
    for i in results:
        f = sp.lambdify((x, y, c, k, a, m, n), i, 'numpy')
        funcs.append(f)
    
    x.evalf()
    y.evalf()
    dVdt = ddtenergy((x, y, c, k, a, m, n, t))
    f = sp.lambdify((x, y, c, k, a, m, n), dVdt, 'numpy')
    
    c = random.randrange(0, 10, 1)
    k = random.randrange(0, 10, 1)
    a=random.randrange(0,10,1)
    m=random.randrange(0,10,1)
    n=random.randrange(0,10,1)
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Vxy = energyfunc((X, Y, c, k, a, m, n))
    plotting(X, Y, Vxy, 'V(x, y)', 'Lyapunov Function, V(x, y)', 'Contour plot of V(x, y)', sol, -1)
    
    for i in range(l):
        xminval=(sol[i][0]-0.1)
        xmaxval=(sol[i][0]+0.1)
        
        yminval=(sol[i][1]-0.1)
        ymaxval=(sol[i][1]+0.1)
        
        x = np.linspace(xminval, xmaxval, 400)
        y = np.linspace(yminval, ymaxval, 400)
        X, Y = np.meshgrid(x, y)
        Z=f(X, Y, c, k, a, m, n)
        plotting(X, Y, Z, r'$\frac{d}{dt} V(x,y)$' , 'Surface plot of '+r'$\frac{d}{dt} V(x,y)$', 'Contour plot of'+r'$\frac{d}{dt} V(x,y)$', sol, i)
      
        
    print("\nThe Equilibrium point(s):")
    for i in range(l):
        ans=funcs[i](X, Y, c, k, a, m, n)
        print(f'V{sol[i]} = ', ans)
main()
