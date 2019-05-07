import numpy as np
import matplotlib.pyplot as plotlib

data=np.loadtxt("ex1data2.txt",delimiter=",")
(m,k)=data.shape
parameters=data[:,:k-1]
house_prices=data[:,k-1:k]
prices=data[:,k-1:k]
mu=np.zeros([k-1,1])
sigma=np.zeros([k-1,1])

def feature_scaling(X):
    mean=np.mean(X)
    stand_devtn=np.std(X)
    return (((X-(mean*np.ones([m,1])))/stand_devtn),mean,stand_devtn)

for i in range(k-1):
    (parameters[:,i:i+1],mu[i][0],sigma[i][0])=feature_scaling(parameters[:,i:i+1])
#(house_prices[:,:1],me,sig)=feature_scaling(house_prices[:,:1])

#print(parameters[:5,:])
#plotlib.scatter(parameters[:,:1],parameters[:,1:2],color="red",marker="*")
#plotlib.xlabel("house_size")
#plotlib.ylabel("num_of_bedrooms")
#plotlib.show()
parameters=np.c_[np.ones([m,1]),parameters]
theta=np.zeros([k,1])
num_iters=50

def computeCost(X,Y,theta,m):
    error_matrix= (X @ theta) - Y
    return (error_matrix.transpose() @ error_matrix).item()/(2*m)

def gradientDescentMulti(X,Y,alpha,num_iters,m,num_features,theta):
    cost_func_matrix=np.zeros([num_iters,1])
    for j in range(num_iters):
        theta = theta - ((alpha/m) * (((X @ theta) - Y).transpose() @ X).transpose())
        cost_func_matrix[j][0]=computeCost(X,Y,theta,m)
    return (theta,cost_func_matrix)

def normal_equation(X,Y):
    return np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y)

J_n=list()
learning_rate=[0.3,0.1,0.03,0.01]
for i in range(4):
    (theta_1,J)=gradientDescentMulti(parameters,house_prices,learning_rate[i],num_iters,m,k,theta)
    print(theta_1)
    J_n.append(J)
iters_matrix=np.arange(num_iters)
for i in range(4):
    plotlib.plot(iters_matrix,J_n[i],label=("aplha="+str(learning_rate[i])))

plotlib.xlabel("num_of_iterations")
plotlib.ylabel("Cost_function")
plotlib.legend()
plotlib.show()

#print(theta_1)
theta_2=normal_equation(np.c_[np.ones([m,1]),data[:,:k-1]],house_prices)
print(theta_2)
