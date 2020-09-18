import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from ratfit_exact import rat_fit, rat_eval

n = 10 #number of points
x = np.linspace(-np.pi/2,np.pi/2,100) #For the plot
y = np.cos(x) #For the plot
xp = np.linspace(-np.pi/2,np.pi/2,n) #The number of points being used
yp = np.cos(xp) #Cosine values for these points
plt.plot(xp,yp, "x")
plt.plot(x,y,label="cos(x)")

#Polynomial interpolation (we shall use 10 points for each interpolation i.e. a polynomial of order n-1=9)
ord = n-1
coeffs = np.polyfit(x,y,deg=ord)
ypoly = np.polyval(coeffs, x)
coeffsp= np.polyfit(xp,yp,deg=ord)
yppoly = np.polyval(coeffsp, xp)
plt.plot(x,ypoly,label="Polynomial")
print("Polynomial Std=",np.std(ypoly-y, ddof=1))

#Cubic Spline
spln = interpolate.splrep(xp,yp)
yspln = interpolate.splev(x,spln)
plt.plot(x,yspln,label="Cubic Spline")
print("Cubic Spline Std=",np.std(yspln-y, ddof=1))

#Rational Function (using the functions from lectures)
n,m = 5,6 #as the function is concave down, m>n.
p,q = rat_fit(xp,yp,n,m)
pred = rat_eval(p,q,xp)
yrat = rat_eval(p,q,x)
plt.plot(x, yrat, label="Rational")
print("Rational Std=", np.std(yrat-y, ddof=1))

plt.legend()
plt.show()