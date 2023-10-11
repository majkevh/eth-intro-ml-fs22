#Import packages
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBRegressor

#Remove warning to read the output
warnings.filterwarnings("ignore")

#Import data
train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')
test_data = pd.read_csv('test_features.csv')


#Defining labels
labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
          'LABEL_Bilirubin_total', 'LABEL_Lactate', 
          'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']
regression = ['LABEL_RRate','LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']


#Functions
def transform(matrix):
    number_features = 8 
    temp = []
    for i in range(int(matrix.shape[0] / 12)): #12 h  
        patient_data = matrix[(12 * i):(12 * (i + 1)), 2:]
        features_values = np.zeros((number_features, matrix[:, 2:].shape[1])) 
        
        #Features = mean,  var, median, max, min, first, last#
        features_values[0] = np.nanmean(patient_data, axis=0)
        features_values[1] = np.nanvar(patient_data, axis=0)
        features_values[2] = np.nanmedian(patient_data, axis=0)
        features_values[4] = np.nanmax(patient_data, axis=0)
        features_values[5] = np.nanmin(patient_data, axis=0)
        features_values[6] = patient_data[0, :]
        features_values[7] = patient_data[11, :]
        ######################################################
        
        
        temp.append(features_values.ravel())
    return temp


def identifiers(matrix):
    return matrix[0::12, 0]

#Transform data and go to numpy
x_train = transform(train_features.to_numpy())
x_test = transform(test_data.to_numpy())
y_train = train_labels[labels].to_numpy()
y_reg = train_labels[regression].to_numpy()

#Results and parameter setting
number_features = len(y_train[0])
number_patient_test = int(len(x_test))
results = np.zeros((number_patient_test, 16))


######################## TASK 1 + 2 ###############################
#MODEL
model = make_pipeline(SimpleImputer(strategy='most_frequent'), 
                      StandardScaler(), HistGradientBoostingClassifier())
#TRAIN
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=True)
for i in range(number_features):
    model = model.fit(X_train, Y_train[:, i])
    print("Score " + str(labels[i]) + ":", metrics.roc_auc_score(Y_val[:, i], model.predict_proba(X_val)[:, 1]))
    results[:,i+1] = model.predict_proba(x_test)[:, 1] 

####################### TASK 3 #################################
#MODEL
regressor = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), 
                         XGBRegressor(objective ='count:poisson',max_depth=3, use_label_encoder =False))
#TRAIN
X_train, X_val, Y_train_reg, Y_val_reg = train_test_split(x_train, y_reg, test_size=0.10, shuffle=True)
for i in range(4):
    regressor.fit(X_train, Y_train_reg[:, i])
    y_pred = regressor.predict(X_val)
    accuracy = np.mean(0.5 + 0.5 * np.maximum(0, metrics.r2_score(Y_val_reg[:, i], y_pred)))
    print("Regression " + str(regression[i]) + ":" + str(accuracy))
    results[:, 12+i] = regressor.predict(x_test)


################ COMPRESSION AND RESULTS ##################
results[:, 0] = identifiers(pd.DataFrame.to_numpy(test_data))
results=pd.DataFrame(results, columns=['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct','LABEL_EtCO2', 'LABEL_Sepsis', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2','LABEL_Heartrate'])
results.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

















