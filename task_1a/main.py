import numpy as np
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
from multiprocessing import Pool


#TRAINING
def train(data):
    number_points=len(data)

    np.random.shuffle(data) #shuffle the data (comment out if not needed)

    #identifier=data[:,0]
    y=data[:,0]
    x=data[:,1:14]


    x_CV = np.zeros((int(number_points/10), 13, 10))
    y_CV = np.zeros((int(number_points/10), 10))
    RMSE = np.zeros((5, 10))
    
    for i in range(10):
        x_CV[:, :, i] = x[i*int(number_points/10):((i+1)*int(number_points/10)), :]
        y_CV[:, i] = y[i*int(number_points/10):((i+1)*int(number_points/10))]
        
    weights=np.zeros((13,10, 5))#number x , 10 subìdivision of dataset, 5 lambda 
    
    lambda_CV = np.array([0.1, 1, 10, 100, 200])
    
    for Lambda in range(5):
        A = np.zeros((int(number_points/10)*9, 13))
        y_assembled = np.zeros((int(number_points/10)*9))
        for k_CV in range(10):
            l=0
            for i in range(10):           
                if i != k_CV:
                   A[l*int(number_points/10):(l+1)*int(number_points/10),:]=x_CV[:,:,i]
                   y_assembled[l*int(number_points/10):(l+1)*int(number_points/10)] = y_CV[:, i]
                   l=l+1
            
            weights[:, k_CV, Lambda] = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+ lambda_CV[Lambda]*np.identity(13)), A.T), y_assembled)
            y_model = np.dot(x_CV[:, :, k_CV], weights[:, k_CV, Lambda])
            RMSE[Lambda, k_CV] = mean_squared_error(y_CV[:, k_CV], y_model)**0.5
          #  print(RMSE)
            
           # plt.scatter(y_CV[:, k_CV],y_model)
           # plt.show()
            
            
    
    Solution = RMSE.mean(axis = 1)
    return Solution
    

file=open("train.csv")
data=np.loadtxt(file, delimiter=',', skiprows=1)


repetitions=64 #number of repetitions, multiplied by the number of cpu threads


cpus = mp.cpu_count()
data_array=list()
for i in range (cpus):
    data_array.append(data)
print("Number of available threads: "+ str(cpus))
print("This program is going to repeat training for "+str(repetitions*cpus)+" times (change the value of the variable repetitions to change this)")
print("The data will be shuffled randomly every time (to change this comment out np.random.shuffle(data) from the train function)")
print("Estimated time to completion: " + str(np.round((0.0427*repetitions),1))+" minutes (@3.7 Ghz)")
if 0.0427*repetitions>10:
    print("WARNING: this is going to take a loooong time")
Solution=np.zeros((repetitions*cpus,5))


for q in range(repetitions):
    
    print(str(q*cpus) +" out of "+ str(repetitions*cpus)+" completed")
    #open files
    if __name__ == '__main__':
        with Pool(processes=cpus) as p:
            result=p.map(train, data_array)
            result=np.array(result)
            #print(result)
            Solution[q*cpus:(q+1)*cpus,:]=result
    
Solution_m=np.mean(Solution, axis=0)
print(Solution)
print(Solution_m)

filename="solution.csv"
np.savetxt(filename, Solution_m, delimiter=",")