import numpy as np
import matplotlib.pyplot as plotlib

data=np.loadtxt("data1.txt",delimiter=",")
population=data[:,:1]
(m,k)=population.shape
profit=data[:,1:2]
plotlib.scatter(population,profit,color="red",marker="*")
plotlib.xlabel("population in 10,000s")
plotlib.ylabel("profit in 10,000$")
plotlib.title("Food Truck Profit")
#plotlib.show()

population_matrix=np.c_[np.ones([m,1]),population]
alpha=0.01
epoch=1500
theta=np.zeros([2,1])

def computeCost(X,Y,theta,m):
    return np.sum(((X @ theta)-Y)**2)/(2*m)

print(computeCost(population_matrix,profit,theta,m))

def gradientDescent(X,Y,theta,alpha,num_iters,m):
    for i in range(num_iters):
        theta=theta - ((alpha/m) * (((X @ theta) - Y).transpose() @ X).transpose())
        ofile.write(str(computeCost(X,Y,theta,m)) + '\n')
    return theta

def predict(x,theta):
    return (np.array([[1,x]]) @ theta).item()


ofile=open("CostFunc.txt",'a')
linear_fit=gradientDescent(population_matrix,profit,theta,alpha,epoch,m)
ofile.close()
print(predict(3.5,linear_fit))
print(predict(7,linear_fit))
predicted_profit=np.empty([0,1])
for j in population:
    predicted_profit=np.append(predicted_profit,predict(j.item(),linear_fit))
plotlib.plot(population,predicted_profit)
plotlib.show()
