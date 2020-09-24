import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

"""
To solve for the decay products of U238, we copy the code used in class for the implicit case. The only adjustment that will need to be made is to include more half lives in the evaluation. Each decap step (besides the first and last) will have to include the previous product decaying into it, and the current product decaying into something else. i.e. we should write a for loop to evaluate dydt at each of these times (where x in this case is time, so more like dydt). 
"""
half_lives = [4.468e9, 24.10/365.25, 6.70/(24*365.25), 245500, 75380, 1600, 3.8235/365.25, 3.10/(60*24*365.25), 26.8/(60*24*365.25), 19.9/(60*24*365.25),164.3/((1e-6)*3600*24*365.25), 22.3, 5.015, 138.376/(365.25)] #Written in years

def radioactivity(t,y,half_life=half_lives): #To get the decay, we must muktiply by ln(2) to get the proper decay. We can plot the results and we see that after a period = T1/2, the amount of the substance has dropped to half, as expected. 
    dydt=np.zeros(len(half_life)+1)
    dydt[0] = -(y[0]/half_life[0])
    for i in range(1,len(half_life)):
        dydt[i] = (y[i-1]/half_life[i-1] - y[i]/half_life[i])
    dydt[-1]=y[-2]/half_life[-1]
    return dydt*np.log(2)

y0 = np.zeros(len(half_lives)+1); y0[0] = 1
t = [0,0.5e7]
times = np.linspace(min(t),max(t),10000)
ans_stiff=integrate.solve_ivp(radioactivity,t,y0,method='Radau',t_eval=times)

#Plotting ratio of Pb206 to U238, Pb206 is the last part of the decay chain.
#plt.plot(ans_stiff.t,ans_stiff.y[-1]/ans_stiff.y[0])
#plt.ylabel(r"$\dfrac{\mathrm{^{206}Pb}}{\mathrm{^{238}U}}$", fontsize=12)

#Plotting ratio of Th230 to U234, Th230 is the 5th element in the chain and U234 is the 4th element in the chain.
plt.plot(ans_stiff.t,ans_stiff.y[4]/ans_stiff.y[3])
plt.ylabel(r"$\dfrac{\mathrm{^{230}Th}}{\mathrm{^{234}U}}$", fontsize=12)

plt.xlabel("Time (years)", fontsize=12)
plt.savefig("problem_3_Th.png", bbox_inches="tight", dpi=500)
plt.show()