import numpy as np
from matplotlib import pyplot as plt
from math import exp, floor, log, tan, pi
import scipy.special as sp
from playsound import playsound

def wasserstein(quant, sample):
    sample.sort()
    w = 0
    lower = 0
    upper = len(quant)-1
    l_bound = quantile(0.05)
    u_bound = quantile(0.95)
    while quant[lower] < l_bound or quant[upper] > u_bound:
        if quant[lower] < l_bound:
            lower += 1
        if quant[upper] > u_bound:
            upper -= 1
            
    quant = quant[lower:(upper+1)]
    sample = sample[lower:(upper+1)]
    for i in range(len(quant)):
        w += min([abs(quant[i] - sample[i]),100])**2
        
    w = w/len(quant)
    return w ** (1/2)

#%%
# Laplace distribution
# mu = 0, b = 1

def f_value(v,sigma):
    k = (2-d)/2
    temp = np.matmul(np.matmul(v.transpose(),np.linalg.inv(sigma)),v)
    f = 2/((2*pi)**(d/2) * np.linalg.det(sigma)**0.5)
    return -log((f * (temp/2)**(k/2) * sp.kv(k,(2*temp)**0.5))[0,0])

b = 1/(2**0.5)

def quantile(v):
    if v < 0:
        temp = exp(v/b)/2
        return log(2*temp)*b
    else:
        temp = 1 - exp(-v/b)/2
        return -log(2-2*temp)*b
    
batch_size = 200
scale = 1

def ref(v,sigma):
    noise = np.random.normal(0,1,(d,batch_size))
    s = 0
    for i in range(batch_size):
        s += f_value(v+scale*noise[:,i].reshape((d,1)),sigma)
    return s/batch_size

def ref_grad(v,sigma):
    noise = np.random.normal(0,1,(d,batch_size))
    s = np.zeros((d,1))
    for i in range(batch_size):
        s += f_value(v+scale*noise[:,i].reshape((d,1)),sigma)*noise[:,i].reshape((d,1))
    return s*1/(batch_size*scale)

#%%

sample_size = 500
d = 3
mag = 2

iterations = 300
step_size = 0.05

vects1 = []
vects2 = []
vects3 = []

for _ in range(sample_size):
    vec = np.random.normal(0,mag,(d,1))
    vects1.append(vec)
    vects2.append(vec)
    vects3.append(vec)

q_vect = []
# make the quantile vectors for comparison in Wasserstein distance
for i in range(1,sample_size+1):
    q_vect.append(quantile(i/sample_size))

w1 = []
w2 = []
w3 = []

wass1 = []
wass2 = []
wass3 = []

for i in range(d):
    wass1.append(wasserstein(q_vect,[vects1[j][i,0] for j in range(sample_size)]))
    wass2.append(wasserstein(q_vect,[vects2[j][i,0] for j in range(sample_size)]))
    wass3.append(wasserstein(q_vect,[vects3[j][i,0] for j in range(sample_size)]))
    
w1.append(np.mean(wass1))
w2.append(np.mean(wass2))
w3.append(np.mean(wass3))

#var1 = np.random.normal(1,1,(d,d))
var1 = np.identity(d)
var1[0,1] = 0.5
var1[1,0] = 0.5
var2 = var1
var3 = var1


for z in range(iterations):
    print("_______________________________current iteration is: ", z)
    #var1 = np.cov(np.asarray(vects1).reshape((runs,d)).transpose())
    #var2 = np.cov(np.asarray(vects2).reshape((runs,d)).transpose())
    #var3 = np.cov(np.asarray(vects3).reshape((runs,d)).transpose())
    
    for i in range(sample_size):
    
        # sample the noise
        W = np.random.normal(0,1,(d,1))
        # Overdamped Langevin
        #vects1[i] = vects1[i] - step_size*grad(vects1[i]) + W * (2*step_size)**(1/2)
        vects1[i] = vects1[i] - step_size*ref_grad(vects1[i],var1) + W * (2*step_size) ** (1/2)

        # Multiplicative Langevin
        
        vects2[i] = vects2[i] - step_size*ref_grad(vects2[i],var2)*exp((f_value(vects2[i],var2)-ref(vects2[i],var2))) \
            + W * exp((f_value(vects2[i],var2) - ref(vects2[i],var2))/2)*(2*step_size)**(1/2)

        # Time-change Langevin
        # Run 1 step ODE for the time change l(t)
        print(f_value(vects3[i],var3) - ref(vects3[i],var3))

        subiter = floor(exp((f_value(vects3[i],var3) - ref(vects3[i],var3))))
        leftover = (exp((f_value(vects3[i],var3) - ref(vects3[i],var3))) - subiter)*step_size
        if leftover < 0:
            subiter = subiter - 1
            leftover = leftover + step_size
        
        # Run Langevin for Z
        if subiter > 0:
            for _ in range(subiter):
                temp_noise = np.random.normal(0,1,(d,1))*((2*step_size)**(1/2))
                vects3[i] = vects3[i]-step_size*ref_grad(vects3[i],var3) + temp_noise
        temp_noise = np.random.normal(0,1)*((2*leftover)**(1/2))
        vects3[i] = vects3[i]-leftover*ref_grad(vects3[i],var3) + temp_noise
        
    wass1 = []
    wass2 = []
    wass3 = []

    for i in range(d):
        wass1.append(wasserstein(q_vect,[vects1[j][i,0] for j in range(sample_size)]))
        wass2.append(wasserstein(q_vect,[vects2[j][i,0] for j in range(sample_size)]))
        wass3.append(wasserstein(q_vect,[vects3[j][i,0] for j in range(sample_size)]))
        
    w1.append(np.mean(wass1))
    w2.append(np.mean(wass2))
    w3.append(np.mean(wass3))
        

# plot the wasserstein distance

x = [i for i in range(iterations+1)]
plt.plot(x, w1, label = 'Overdamped')
plt.plot(x, w2, label = 'Multiplicative')
plt.plot(x, w3, label = 'Time-change')
plt.ylabel('Wasserstein distance')
plt.xlabel('Iteration')
plt.legend()
plt.show()

playsound('FFVictory.wav', False)
        
