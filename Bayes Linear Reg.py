import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from math import exp, floor

# define the metric for methods comparison
def wasserstein(mean, var):
    distance = np.sum(np.square(mean - m))
    mat = sqrtm(var) - sqrtm(V)
    
    distance += np.sum(np.sum(np.square(mat)))
    return distance**(1/2)

#%%
# define the loss and Gaussian smoothing as the reference function
def f_value(v):
    return (np.sum(np.square(np.matmul(v.transpose(),X)-y))+np.sum(np.square(v))/(2*(mag**2)))

def grad(v):
    return (2*np.sum(np.matmul(v.transpose(),X)-y)+1/(mag**2))*v

batch_size = 200
scale = 0.04

def ref(v):
    noise = np.random.normal(0,1,(d,batch_size))
    s = 0
    for i in range(batch_size):
        s += f_value(v+scale*noise[:,i].reshape((d,1)))
    return s/batch_size

def ref_grad(v):
    noise = np.random.normal(0,1,(d,batch_size))
    s = np.zeros((d,1))
    for i in range(batch_size):
        s += f_value(v+scale*noise[:,i].reshape((d,1)))*noise[:,i].reshape((d,1))
    return s*1/(batch_size*scale)

#%%
sample_size = 200
d = 3
mag = 5

# create synthetic data from normal distribution
noise = np.random.normal(0,1,(1,sample_size))

X = np.random.normal(0,1,(d,sample_size))
vec = np.random.normal(1,mag,(d,1))
y = np.matmul(vec.transpose(),X) + noise

m = np.matmul(np.linalg.inv(np.eye(d) * (1/mag) + np.matmul(X,X.transpose())),np.matmul(X,y.transpose())).reshape(d)
V = np.linalg.inv(np.eye(d) * (1/mag) + np.matmul(X,X.transpose()))

# run Bayesian linear regression

iterations = 100
step_size = 0.00005

runs = 100

vects1 = [vec.copy() for _ in range(runs)]
vects2 = [vec.copy() for _ in range(runs)]
vects3 = [vec.copy() for _ in range(runs)]

w1 = []
w2 = []
w3 = []
mean = np.mean(np.asarray(vects1).reshape((runs,d)), axis = 0)
var = np.cov(np.asarray(vects1).reshape((runs,d)).transpose())
w1.append(wasserstein(mean,var))
w2.append(wasserstein(mean,var))
w3.append(wasserstein(mean,var))

for z in range(iterations):
    print("_______________________________current iteration is: ", z)
    for i in range(runs):
    
        # sample the noise
        W = np.random.normal(0,1,(d,1))
        # Overdamped Langevin
        #vects1[i] = vects1[i] - step_size*grad(vects1[i]) + W * (2*step_size)**(1/2)
        vects1[i] = vects1[i] - step_size*ref_grad(vects1[i]) + W * (2*step_size) ** (1/2)

        # Multiplicative Langevin
        
        vects2[i] = vects2[i] - step_size*ref_grad(vects2[i])*exp(f_value(vects2[i])-ref(vects2[i])) \
            + W * exp((f_value(vects2[i]) - ref(vects2[i]))/2)*(2*step_size)**(1/2)

        # Time-change Langevin
        # Run 1 step ODE for the time change l(t)
        print(f_value(vects3[i]) - ref(vects3[i]))

        subiter = floor(exp(f_value(vects3[i]) - ref(vects3[i])))
        leftover = (exp(f_value(vects3[i]) - ref(vects3[i])) - subiter)*step_size
        if leftover < 0:
            subiter = subiter - 1
            leftover = leftover + 1
        
        # Run Langevin for Z
        if subiter > 0:
            for _ in range(subiter):
                temp_noise = np.random.normal(0,1,(d,1))*((2*step_size)**(1/2))
                vects3[i] = vects3[i]-step_size*ref_grad(vects3[i]) + temp_noise
        temp_noise = np.random.normal(0,1)*((2*leftover)**(1/2))
        vects3[i] = vects3[i]-leftover*ref_grad(vects3[i]) + temp_noise


    mean = np.mean(np.asarray(vects1).reshape((runs,d)), axis = 0)
    var = np.cov(np.asarray(vects1).reshape((runs,d)).transpose())
    w1.append(wasserstein(mean,var))
    
    mean = np.mean(np.asarray(vects2).reshape((runs,d)), axis = 0)
    var = np.cov(np.asarray(vects2).reshape((runs,d)).transpose())
    w2.append(wasserstein(mean,var))
    
    mean = np.mean(np.asarray(vects3).reshape((runs,d)), axis = 0)
    var = np.cov(np.asarray(vects3).reshape((runs,d)).transpose())
    print("var is: ", var)
    w3.append(wasserstein(mean,var))
    
# plot the metric of the three methods and compare

xvec = [i for i in range(iterations+1)]
plt.plot(xvec, w1, label = 'Overdamped')
plt.plot(xvec, w2, label = 'Multiplicative')
plt.plot(xvec, w3, label = 'Time-change')
plt.ylabel('Wasserstein distance')
plt.xlabel('Iteration')
plt.legend()
plt.show()
