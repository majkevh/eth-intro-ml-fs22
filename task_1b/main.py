import numpy as np


#TRAINING
#open files
file=open("train.csv")
data=np.loadtxt(file, delimiter=',', skiprows=1)


number_points=len(data)

#identifier=data[:,0]
y=data[:,1]
x=data[:,2:7]

weights=np.zeros((21,1))

A=np.ones((number_points,21))

for i in range(21):
    if (i  < 5):
        A[:, i] = x[:, i]
    elif (5 <= i <= 9):
        A[:, i] = x[:, i-5]**2
    elif (10 <= i <= 14):
        A[:, i] = np.exp(x[:, i-10])
    elif (15<= i <= 19):
        A[:, i] = np.cos(x[:, i-15]) 
        
inverse = np.linalg.inv(np.matmul(A.T,A))
weights = np.matmul(np.matmul(inverse,A.T),y)#(A^T A)^{-1}A^T y
    #print(weights)


filename="solution.csv"
np.savetxt(filename, weights, delimiter=",")