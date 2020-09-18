import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from ratfit_exact import rat_fit, rat_eval

nps = 5 #number of points
x = np.linspace(-1,1,100) #For the plot
y = 1/(1+x**2) #For the plot
xp = np.linspace(-1,1,nps) #Number of points being used
yp = 1/(1+xp**2) #Values at these points

#Polynomial interpolation (we shall use nps points for each interpolation, however to match the order of rational, we have
#ord = nps+1). 
ord = nps+1
coeffs = np.polyfit(xp,yp,deg=ord)
ypoly = np.polyval(coeffs, x)
plt.plot(x,y-ypoly,label="Polynomial")
print("Polynomial Std=",np.std(ypoly-y, ddof=1))

#Cubic Spline
spln = interpolate.splrep(xp,yp)
yspln = interpolate.splev(x,spln)
plt.plot(x,y-yspln,label="Cubic Spline")
print("Cubic Spline Std=",np.std(yspln-y, ddof=1))

#Rational Function (using the functions from lectures)
n,m = int(nps/2) + 1, int(nps/2) #as the function is concave down, m>n, n+m-1 = nps.
p,q = rat_fit(xp,yp,n,m)
yrat = rat_eval(p,q,x)
plt.plot(x,y-yrat, label="Rational")
print("Rational Std=", np.std(yrat-y, ddof=1))

plt.legend()
plt.savefig("lor_interpolation.png", bbox_inches="tight", dpi=500)
plt.show()