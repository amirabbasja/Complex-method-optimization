################################
#    Code by Abbas jafarpour
################################
import numpy as np
import random

#Get necessary variables
e = 0.00001 #Termination criterion
n = 2 #Number of variables
alpha = 1.3 #Reflection coefficeint
alphaMinVal = 0.001 #Min value for alpha, if reached we will move onto next Woest point
varBound = [-3 ,3 ,-3 ,3] #Boundary of variables e.g. [lowerBoundOfX1 upperBoundOfX1 lowerBoundOfX2 upperBoundOfX2 ...]

#Defining the constraints
def constraints(x):
    #All the constraints are defined lower than zero
    #if all constraints are met, return True, else return False
    const = []
    const = const[:] + [-x[0]**3 - 2*x[1]**3 - 2*x[0]**2 - x[1]**2 + x[0]*x[1] + 4]

    #If even one of constraints are bigger than 0, this function will return False
    for i in range(len(const)):
        if (0 < const[i]):
            return False

    return True

def objective(x):
    #All the constraints are defined lower than zero
    obj = []
    obj = obj[:] + [7*x[0]*x[1]/np.exp(x[0]**2+x[1]**2-1)] 
    return obj

def centroid(x,subtractXH,secondXH = False):
    #Finding the centroid of all nodes except xh
    #subtractXH is a bool and if true, xh will be calculated and will be subtracted from sum
    sum = [0 , 0]
    n = len(x)
    sum = np.sum(x,axis = 0)
    
    if subtractXH:
        if(secondXH):
            xh, _ = GetSecondXH(x)
        else:
            xh, _ = GetXH(x)
        sum = np.subtract(sum, xh)
        return np.divide(sum,n-1)
    else:
        return np.divide(sum,n)

def GetXH(x):
    #Finding the worst node
    f = []
    for i in range (len(x)):
        f = f[:] + [objective(x[i])]
    
    XHindex = f.index(max(f))

    #Returns xh, xRest
    return x[XHindex] , x[:XHindex] + x[XHindex+1:]

def GetXR(x, alpha):
    #Find the weighted reflection
    xh, xrest = GetXH(x)
    xc = centroid(x, True)
    return [x + y for (x, y) in zip(xc,alpha*(np.subtract(xc,xh)))]

def TerminationCriterion(x,e):
    #Stopping criterion. if met, send true and terminate
    #Else, return false and continue the search
    sum = 0
    n = int(len(x)/2)
    xcc = centroid(x, True)

    for i in range(n):
        sum += np.absolute(np.subtract(objective(x[i]), objective(xcc)))

    if sum/2/n <= e:
        return True
    else:
        return False

def GetSecondXR(x, alpha):
    #Find the weighted reflection
    xh, xrest = GetSecondXH(x)
    xc = centroid(x, True, True)
    return [x + y for (x, y) in zip(xc,alpha*(np.subtract(xc,xh)))]

def GetSecondXH(x):
    
    f = []
    for i in range (len(x)):
        f = f[:] + [objective(x[i])]

    mx=max(f[0],f[1])
    secondmax=min(f[0],f[1])
    n =len(f)
    for i in range(2,n):
        if f[i]>mx:
            secondmax=mx
            mx=f[i]
        elif f[i]>secondmax and \
            mx != f[i]:
            secondmax=f[i]
    
    XH2index = f.index(secondmax)

    #Returns xh2, xRest
    return x[XH2index] , x[:XH2index] + x[XH2index+1:]

#Generate 2n feasible points
def GenerateRandomStartNumbers(varBound):
    restart = False
    while(True):
        if(not restart):
            #Generate first point, we have used an infinite while loop to get the first feasible point no matter what 
            while (True and not restart):
                x1 = [random.uniform(varBound[0], varBound[1]),random.uniform(varBound[2], varBound[3])]
                if (constraints(x1)):
                    break

            #Generate the second feasible point, but if the random point is not feasible;
            #we approach the previouse feasible location by half the current distance of these points
            x2 = [random.uniform(varBound[0], varBound[1]),random.uniform(varBound[2], varBound[3])]
            while (True and not restart):
                if (constraints(x2)):
                    x2New = np.mean([centroid([x1],False) , x2], axis = 0)
                    if(np.absolute(np.subtract(x2,x2New)[0])<0.01):
                        restart = True
                    else:
                        x2 = np.ndarray.tolist(x2New)
                else:
                    break

            x3 = [random.uniform(varBound[0], varBound[1]),random.uniform(varBound[2], varBound[3])]
            while (True and not restart):
                if (constraints(x3)):
                    x3New = np.mean([centroid([x1, x2],False) , x3], axis = 0)
                    if(np.absolute(np.subtract(x3,x3New)[0])<0.01):
                        restart = True
                    else:
                        x3 = np.ndarray.tolist(x3New)
                    
                else:
                    break

            x4 = [random.uniform(varBound[0], varBound[1]),random.uniform(varBound[2], varBound[3])]
            while (True and not restart):
                if (constraints(x4)):
                    x4New = np.mean([centroid([x1, x2, x3],False) , x4], axis = 0)
                    if(np.absolute(np.subtract(x4,x4New)[0])<0.01):
                        restart = True
                    else:
                        x4 = np.ndarray.tolist(x4New)
                    
                else:
                    break

            return [x1,x2,x3,x4]
        else:
            restart = False
x = GenerateRandomStartNumbers(varBound)

#Start the searching process
j = 0
k = 0
secondWorst = False

#Coordinates of XR at all times
xr_x = []
xr_y = []

while (True):
    #Get XR, XH and XC; Also if secondWorst is True, it means that alpha has been so little and we have to lose the last
    #Worst node and go to the second worst node
    if(secondWorst):
        XR = GetSecondXR(x,alpha)
        XH, restOfx = GetSecondXH(x)
        XC = centroid(x, True, True)
    else:
        XR = GetXR(x,alpha)
        XH, restOfx = GetXH(x)
        XC = centroid(x, True)

    #Check to see if XC is feasible, if it is not, start over with new beginning points
    if((varBound[0] <= XC[0] and XC[0] <= varBound[1]) and (varBound[2] <= XC[1] and XC[1] <= varBound[3])):
        x = GenerateRandomStartNumbers(varBound)

    #Check the feasibility of XR or f(XR) < f(XH)
    if (constraints(XR) and (objective(XR) < objective(XH))):
        #Replace XH with XR
        xr_x.append(XR[0])
        xr_y.append(XR[1])
        
        #Replace XH with XR
        x = restOfx[:] + [XR]
        secondWorst = False
        if(TerminationCriterion(x,e)):
            print("Local minimum found! x = {}  |  f(x) = {}".format(x[0],objective(x[0])))
            break
    #If conditions are not met
    else:
        k = 0
        while (True):
            a = alpha/2*(0.5**k)
            k = k + 1
            if (alphaMinVal < a):
                XR = [x + y for (x, y) in zip(XC,a*(np.subtract(XC,XH)))] 
                if(constraints(XR) and (objective(XR) < objective(XH))):
                    x = restOfx[:] + [XR]
                    break
                secondWorst = False
            else:
                secondWorst = True
                break
                
from scipy.optimize import Bounds , LinearConstraint , NonlinearConstraint , minimize , BFGS
import matplotlib.pyplot as plt
import numpy as np

print(xr_x)
plt.scatter(np.array(xr_x), np.array(xr_y), s = 2)
plt.show()

#Define the objective function
def objective1(x):
    return 7*x[0]*x[1]/np.exp(x[0]**2+x[1]**2-1)

#Defining the costraints
def cons_f(x):
    return [x[0],x[1],-x[0]**3 - 2*x[1]**3 - 2*x[0]**2 - x[1]**2 + x[0]*x[1] + 4]

nonlinearConst = NonlinearConstraint(cons_f,[-3,-3,-np.inf],[3,3,0], 
                                     jac = '2-point', hess = BFGS() )

#Beginning point
x0 = x[0]

res = minimize(objective1,
               x0, method='trust-constr',
               constraints=nonlinearConst,
               jac = '2-point',
               hess = BFGS())
print(res)