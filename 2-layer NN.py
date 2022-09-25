import numpy as np
from math import exp, floor
import pandas as pd
from sklearn.metrics import accuracy_score
#import breast cancer data and change the labels

df = pd.read_csv('data.csv')
Y = df['diagnosis']
X = df.drop(['id','diagnosis'],axis = 1)

sample_size = X.shape[0]
d = X.shape[1]

print(sample_size)
print(d)

for i in range(len(Y)):
    if Y[i] == "B":
        Y[i] = 1
    else:
        Y[i] = 0
        
X = X.to_numpy()
Y = Y.to_numpy(dtype='int')

print(X.shape)
print(Y.shape)

#normalize the features

for i in range(d):
    temp = X[:,i]
    temp = temp - np.min(temp)
    temp = temp/np.max(temp)
    temp = (temp - 0.5)*2
    X[:,i] = temp

# Gaussian Smoothing Logistic Regression

class OriginalRegressor():
    
    def fit(self, x, y, epochs, prior1, prior2):

        self.weights1 = prior1
        self.weights2 = prior2
        self.accuracies = []
        relu_output = []
        for j in range(nnodes):
            x_dot_weights = np.matmul(self.weights1[j], x.transpose())
            relu_output.append(self._relu(x_dot_weights))
        relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
        sigmoid_input = np.matmul(self.weights2, relu_output)
        pred = self._sigmoid(sigmoid_input)
        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        self.accuracies.append(accuracy_score(y, pred_to_class)*100)
    
        for i in range(epochs):
            error_w1, error_w2 = self.compute_gradients(x, y)
            self.update_model_parameters1(error_w1)
            self.update_model_parameters2(error_w2)
            
            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j], x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.accuracies.append(accuracy_score(y, pred_to_class)*100)
    
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])
    
    def _relu(self, x):
        return np.array([np.max([value,0]) for value in x])
    
    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
        
    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = np.real(y_true * np.log(y_pred + 1e-9))
        y_one_loss = np.real((1-y_true) * np.log(1 - y_pred + 1e-9))
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true):
        # derivative of binary cross entropy

        temp1 = [np.zeros((d)) for _ in range(nnodes)]
        temp2 = np.zeros((nnodes))
        for i in range(batch_size):
            noise1 = np.random.normal(0,1,(nnodes,d))
            noise2 = np.random.normal(0,1,(nnodes))
            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j]+scale*noise1[j,:].reshape((d)), x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2+scale*noise2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            loss = self.compute_loss(y_true,pred)
            for j in range(nnodes):
                temp1[j] += loss*noise1[j,:].reshape((d))
            temp2 += loss*noise2
        gradients_w1 = [temp1[j]/(batch_size*scale) for j in range(nnodes)]
        gradients_w2 = temp2/(batch_size*scale)
    
        return gradients_w1, gradients_w2
    
    def update_model_parameters1(self, error_w1):
        for j in range(nnodes):
            self.weights1[j] = self.weights1[j] - step_size * error_w1[j] + np.random.normal(0,1,(d)) * (2*step_size) ** (1/2)
    
    def update_model_parameters2(self, error_w2):
        self.weights2 = self.weights2 - step_size * error_w2 + np.random.normal(0,1,(nnodes)) * (2*step_size) ** (1/2)
    
# Multiplicative Langevin Logistic Regression with Gaussian Smoothing

class MultiplicativeRegressor():
    
    def fit(self, x, y, epochs, prior1, prior2):
        
        self.weights1 = prior1
        self.weights2 = prior2
        self.accuracies = []
        relu_output = []
        for j in range(nnodes):
            x_dot_weights = np.matmul(self.weights1[j], x.transpose())
            relu_output.append(self._relu(x_dot_weights))
        relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
        sigmoid_input = np.matmul(self.weights2, relu_output)
        pred = self._sigmoid(sigmoid_input)
        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        self.accuracies.append(accuracy_score(y, pred_to_class)*100)
    
        for i in range(epochs):

            error_w1, error_w2, ref = self.compute_gradients(x, y)
            self.update_model_parameters(error_w1, error_w2, ref, x, y)

            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j], x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.accuracies.append(accuracy_score(y, pred_to_class)*100)
    
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])
    
    def _relu(self, x):
        return np.array([np.max([value,0]) for value in x])
    
    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
        
    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = np.real(y_true * np.log(y_pred + 1e-9))
        y_one_loss = np.real((1-y_true) * np.log(1 - y_pred + 1e-9))
        return -np.mean(y_zero_loss + y_one_loss)
    
    def compute_gradients(self, x, y_true):
        # derivative of binary cross entropy

        temp1 = [np.zeros((d)) for _ in range(nnodes)]
        temp2 = np.zeros((nnodes))
        ref_temp = 0
        for i in range(batch_size):
            noise1 = np.random.normal(0,1,(nnodes,d))
            noise2 = np.random.normal(0,1,(nnodes))
            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j]+scale*noise1[j,:].reshape((d)), x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2+scale*noise2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            loss = self.compute_loss(y_true,pred)
            for j in range(nnodes):
                temp1[j] += loss*noise1[j,:].reshape((d))
            temp2 += loss*noise2
            ref_temp += loss
        gradients_w1 = [temp1[j]/(batch_size*scale) for j in range(nnodes)]
        gradients_w2 = temp2/(batch_size*scale)
    
        return gradients_w1, gradients_w2, ref_temp/batch_size

    def update_model_parameters(self, error_w1, error_w2, ref, x, y):
        relu_output = []
        for j in range(nnodes):
            x_dot_weights = np.matmul(self.weights1[j], x.transpose())
            relu_output.append(self._relu(x_dot_weights))
        relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
        sigmoid_input = np.matmul(self.weights2, relu_output)
        pred = self._sigmoid(sigmoid_input)
        loss = self.compute_loss(y,pred)
        for j in range(nnodes):
            self.weights1[j] = self.weights1[j] - step_size*error_w1[j]*exp(loss-ref) \
                + np.random.normal(0,1,(d))*exp((loss-ref)/2)*(2*step_size)**(1/2)
    
        self.weights2 = self.weights2 - step_size*error_w2*exp(loss-ref) \
            + np.random.normal(0,1,(nnodes))*exp((loss-ref)/2)*(2*step_size)**(1/2)

# Time-change Langevin Logistic Regression with Gaussian Smoothing

class TimechangeRegressor():

    def fit(self, x, y, epochs, prior1, prior2):
    
        self.weights1 = prior1
        self.weights2 = prior2
        self.accuracies = []
        relu_output = []
        for j in range(nnodes):
            x_dot_weights = np.matmul(self.weights1[j], x.transpose())
            relu_output.append(self._relu(x_dot_weights))
        relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
        sigmoid_input = np.matmul(self.weights2, relu_output)
        pred = self._sigmoid(sigmoid_input)
        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        self.accuracies.append(accuracy_score(y, pred_to_class)*100)
    
        for i in range(epochs):
            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j], x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            loss = self.compute_loss(y,pred)
            error_w1, error_w2, ref = self.compute_gradients(x, y)
            
            self.update_model_parameters(error_w1, error_w2, exp(loss-ref)*step_size)
            
            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j], x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.accuracies.append(accuracy_score(y, pred_to_class)*100)
    
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])
    
    def _relu(self, x):
        return np.array([np.max([value,0]) for value in x])
    
    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
        
    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = np.real(y_true * np.log(y_pred + 1e-9))
        y_one_loss = np.real((1-y_true) * np.log(1 - y_pred + 1e-9))
        return -np.mean(y_zero_loss + y_one_loss)
    
    def compute_gradients(self, x, y_true):
        # derivative of binary cross entropy

        temp1 = [np.zeros((d)) for _ in range(nnodes)]
        temp2 = np.zeros((nnodes))
        ref_temp = 0
        for i in range(batch_size):
            noise1 = np.random.normal(0,1,(nnodes,d))
            noise2 = np.random.normal(0,1,(nnodes))
            relu_output = []
            for j in range(nnodes):
                x_dot_weights = np.matmul(self.weights1[j]+scale*noise1[j,:].reshape((d)), x.transpose())
                relu_output.append(self._relu(x_dot_weights))
            relu_output = np.reshape(np.asarray(relu_output),(nnodes,sample_size))
            sigmoid_input = np.matmul(self.weights2+scale*noise2, relu_output)
            pred = self._sigmoid(sigmoid_input)
            loss = self.compute_loss(y_true,pred)
            for j in range(nnodes):
                temp1[j] += loss*noise1[j,:].reshape((d))
            temp2 += loss*noise2
            ref_temp += loss
        gradients_w1 = [temp1[j]/(batch_size*scale) for j in range(nnodes)]
        gradients_w2 = temp2/(batch_size*scale)
    
        return gradients_w1, gradients_w2, ref_temp/batch_size
    
    def update_model_parameters(self, error_w1, error_w2, step):
        for j in range(nnodes):
            self.weights1[j] = self.weights1[j] - step * error_w1[j] + np.random.normal(0,1,(d)) * (2*step) ** (1/2)
        self.weights2 = self.weights2 - step * error_w2 + np.random.normal(0,1,(nnodes)) * (2*step) ** (1/2)
    
nnodes = 32
step_size = 0.5
scale = 3
iterations = 80
batch_size = 100
runs = 20

def runRegressor():
    
    acc1 = [0] * (iterations+1)
    acc2 = [0] * (iterations+1)
    acc3 = [0] * (iterations+1)
    
    for j in range(runs):
        print("_______________________________current run is: ", j)
        
        pri1 = np.random.normal(0,5,size = (nnodes,d))
        pri2 = np.random.normal(0,5,size = (nnodes))
        
        lr1 = OriginalRegressor()
        lr1.fit(x = X, y = Y, epochs=iterations, prior1=pri1.copy(), prior2=pri2.copy())

        lr2 = MultiplicativeRegressor()
        lr2.fit(x = X, y = Y, epochs=iterations, prior1=pri1.copy(), prior2=pri2.copy())

        lr3 = TimechangeRegressor()
        lr3.fit(x = X, y = Y, epochs=iterations, prior1=pri1.copy(), prior2=pri2.copy())

        acc1 = [(acc1[k]*j + lr1.accuracies[k])/(j+1) for k in range(iterations+1)]
        acc2 = [(acc2[k]*j + lr2.accuracies[k])/(j+1) for k in range(iterations+1)]
        acc3 = [(acc3[k]*j + lr3.accuracies[k])/(j+1) for k in range(iterations+1)]
        
    return acc1, acc2, acc3

result1, result2, result3 = runRegressor()
print(" iterations = ", iterations)
print("result1 = ")
print(result1)
print("result2 = ")
print(result2)
print("result3 = ")
print(result3)