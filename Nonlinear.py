# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:04:51 2022

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import math
import numpy as np
import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 

# Importing the library to generate the Mackey Glass time series
from NonlinearGenerator import Nonlinear

# Defining the atributes
from statsmodels.tsa.tsatools import lagmat

# Importing the model
from eMG import eMG


#-----------------------------------------------------------------------------
# Generating the Nonlinear time series
#-----------------------------------------------------------------------------
    
sample_n = 6000
NTS, u = Nonlinear(sample_n)

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y
    
def Normalize_Data(data):
    Max_data = np.max(data)
    Min_data = np.min(data)
    Normalized_data = (data - Min_data)/(Max_data - Min_data)
    return Normalized_data, Max_data, Min_data
        

# Defining the atributes and the target value
X, y = Create_Leg(NTS, ncols = 2, leg = 1, leg_output = 1)
X = np.append(X, u[:X.shape[0]].reshape(-1,1), axis = 1)

# Spliting the data into train and test
X_train, X_test = X[2:5002,:], X[5002:5202,:]
y_train, y_test = y[2:5002,:], y[5002:5202,:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[2:5002,:], Normalized_X[5002:5202,:]
Normalized_y_train, Normalized_y_test = Normalized_y[2:5002,:], Normalized_y[5002:5202,:]


# Plotting rules' evolution of the model
plt.plot(y_test, color='blue')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.show()

#-----------------------------------------------------------------------------
# Calling the eMG
#-----------------------------------------------------------------------------

# Setting the hyperparameters
alpha = 0.01 # Basic learning rate
lambda1 = 0.1 # Significance level
w = 15 # Window of size
sigma = 0.00001 # Parameter to define the initial dispersion matrix
omega = 10**3 # Large real value [10^2, 10^4]

# Initializing the model
model = eMG(alpha = alpha, lambda1 = lambda1, w = w, sigma = sigma, omega = omega)
# Train the model
OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
# Test the model
OutputTest = model.predict(Normalized_X_test)

#-----------------------------------------------------------------------------
# Evaluate the model's performance
#-----------------------------------------------------------------------------

# Calculating the error metrics
# DeNormalize the results
OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, OutputTestDenormalized)

# Printing the RMSE
print("RMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE = ", MAE)
# Printing the number of final rules
print("Final Rules = ", Rules[-1])

#-----------------------------------------------------------------------------
# Plot the graphics
#-----------------------------------------------------------------------------

# Plot the graphic of the actual time series and the eMG predictions
plt.plot(y_test, label='Actual Value', color='red')
plt.plot(OutputTestDenormalized, color='blue', label='eMG')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plot the evolution of the model's rule
plt.plot(Rules, color='blue')
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()