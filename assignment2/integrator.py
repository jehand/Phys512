import numpy as np

#Old Integrator
def integrate_step(fun,x1,x2,tol): 
    x=np.linspace(x1,x2,5)
    y=fun(x) #only want this called once in the new integrator
    n = len(x) #Add a term n to count the number of times we call fun.
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    print("bas",y)
    if myerr<tol:
        return area2,n
    else:
        xm=0.5*(x1+x2)
        a1,n1=integrate_step(fun,x1,xm,tol/2)
        a2,n2=integrate_step(fun,xm,x2,tol/2)
        return a1+a2,n1+n2+n

"""New Integrator: Using some new adaptation of simpsons method. If we study the previous integrator, we realise that each time the ytion is recursively called, the line y=fun(x) is repeated. This is the line we only want to run once. Essentially, the only thing that changes each time this ytion is called is x1 and x2, hence x changes. xm is the middle of both bounds and hence, we don't need to evaluate our ytion for 5 points each time it is called since we already know what the middle term is, and we also know what the ytion is at the bounds. We would just need to calculate the two intermediate points, i.e. point 1 and 3. Hence, we are not calling f(x) for the same x each time which would be 5 calculations, instead we are doing 2 calculations. To do this, we need a new variable n that stores the number of times the integrator has been called, so that if its been called already it does not calculate y=fun(x) again. We need to adjust the values of the bounds that we use for the ytion though.
"""
def integrator(f,x1,x2,tol,n=0,y="",xm=0,sy=""):
    x = np.linspace(x1,x2,5)
    if n == 0: #If has been evaluated 0 times, then it evaluates f(x)
        y = f(x)
        n = len(x)
    else: #If has been evaluated >0 times, then it just calculates the two new values that it needs to.
        if x1<xm: #Something wrong here, possibly the condition is wrong?
            print(1,y)
            y[4] = y[2] #Last point is messed up for some reason
            y[2] = y[1]
        else:
            print(2,y)
            y[0] = y[2]
            y[2] = y[3]
        y[1] = f(x[1]) #This is fine
        y[3] = f(x[3]) #^
        n = 2
    area1 = (x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2 = (x2-x1)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr = np.abs(area1-area2)
    print("after",y)
    if myerr < tol:
        print("returning",y,sy)
        y=sy
        return area2, n
    else:
        xm = 0.5*(x1+x2)
        sy=y
        a1,n1 = integrator(f,x1,xm,tol/2,n,y,xm,sy)
        a2,n2 = integrator(f,xm,x2,tol/2,n,y,xm,sy)
        return a1+a2, n1+n2+n