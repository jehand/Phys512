import numpy as np

def integrate_step(fun,z,x1,x2,tol):
    x=np.linspace(x1,x2,5)
    y=fun(x,z)
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,z,x1,xm,tol/2)
        a2=integrate_step(fun,z,xm,x2,tol/2)
        return a1+a2