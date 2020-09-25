import numpy as np

#Old Integrator
def integrate_step(fun,x1,x2,tol): 
    x=np.linspace(x1,x2,5)
    y=fun(x) #only want this called once in the new integrator
    n = len(x) #Add a term n to count the number of times we call fun.
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2,n
    else:
        xm=0.5*(x1+x2)
        a1,n1=integrate_step(fun,x1,xm,tol/2)
        a2,n2=integrate_step(fun,xm,x2,tol/2)
        return a1+a2,n1+n2+n

"""New Integrator: Using new version of simpsons method (Adaptive simpsons integrator:
https://rosettacode.org/wiki/Numerical_integration/Adaptive_Simpson%27s_method#Python). If we study the previous integrator, we realise that each time
the function is recursively called, the line y=fun(x) is repeated. This is the line we only want to run once. We don't need to evaluate our function
for 5 points each time it is called since we already know what the middle term is, and we also know what the function is at the bounds. We would just
need to calculate the two intermediate points, i.e. point 1 and 3. Hence, we are not calling f(x) for the same x each time which would be 5
calculations, instead we are doing 2 calculations. To do this, we need a new variable n that stores the number of times the integrator has been
called, so that if its been called already it does not calculate y=fun(x) again. Furthermore, we need to make sure that the function stores the values
of y[0],y[2] and y[4], especially when the first integrator returns, otherwise the second integrator will be run with the incorrect y-array. We can
write these arguments into the function to store each time and update the values, therefore not changing the value stored, only the value used for the
next recursion.
"""

"""
asdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgdasdgasdgasdgasgd
"""

def integrator(f,x1,x2,tol,y0=0,y2=0,y4=0,n=0):
    x = np.linspace(x1,x2,5)
    if n == 0: #If has been evaluated 0 times, then it evaluates f(x) at 5 points
        y0,y1,y2,y3,y4 = f(x) #Defining all the variables so that the function values are stored each iteration
        n = len(x)
    else: #If has been evaluated >0 times, then it just calculates the two new values that it needs to
        y1 = f(x[1]) #Re-calculating the two points on either side of ym that need to be calculated
        y3 = f(x[3])
        n = 2
    area1 = (x2-x1)*(y0+4*y2+y4)/6
    area2 = (x2-x1)*(y0+4*y1+2*y2+4*y3+y4)/12
    myerr = np.abs(area1-area2)
    if myerr < tol:
        return area2, n
    else:
        xm = 0.5*(x1+x2)
        a1,n1 = integrator(f,x1,xm,tol/2,y0,y1,y2,n) #Changing what we use as y1,ym and y2 as mentioned above
        a2,n2 = integrator(f,xm,x2,tol/2,y2,y3,y4,n)
        return a1+a2, n1+n2+n