# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
# Binomial cumulative distribution function
from scipy.stats import binom
# Inverse chi square
from scipy.stats.distributions import chi2
#from scipy.spatial import distance
import pandas as pd
import numpy as np
import math
import mahalanobis_distance
class eMG:
    def __init__(self, alpha = 0.01, lambda1 = 0.1, w = 10, sigma = 0.05, omega = 10^2):
        self.hyperparameters = pd.DataFrame({'alpha':[alpha], 'lambda1':[lambda1], 'w':[w], 'sigma':[sigma], 'omega': [omega]})
        self.parameters = pd.DataFrame(columns = ['Center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Gamma', 'Q', 'LocalOutput'])
        # Defining the initial dispersion matrix
        self.Sigma_init = np.array([])
        # Defining the threshold for the compatibility measure
        self.Tp = None
        # Defining the threshold for the arousal index
        self.Ta = 1 - self.hyperparameters.loc[0, 'lambda1']
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
         
    def fit(self, X, y):
        # Initializing the initial dispersion matrix
        self.Sigma_init = self.hyperparameters.loc[0, 'sigma'] * np.eye(X.shape[1])
        # Initializing the threshold for the compatibility measure
        self.Tp = chi2.ppf(1 - self.hyperparameters.loc[0, 'lambda1'], df=X.shape[1])
        # Initialize the first rule
        self.Initialize_First_Cluster(X[0,], y[0])
        for k in range(X.shape[0]):
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
            # Compute the compatibility measure and the arousal index for all rules
            Output = 0
            sumCompatibility = 0
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i)
                self.Arousal_Index(X[k,], i)
                # Local output
                self.parameters.at[i, 'LocalOutput'] = xk @ self.parameters.loc[i, 'Gamma']
                Output = Output + self.parameters.at[i, 'LocalOutput'] * self.parameters.loc[i, 'CompatibilityMeasure']
                sumCompatibility = sumCompatibility + self.parameters.loc[i, 'CompatibilityMeasure']
            # Global output
            if sumCompatibility == 0:
                Output = 0
            else:
                Output = Output/sumCompatibility
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            # Residual
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            center = -1
            compatibility = -1
            for i in self.parameters.index:
                chistat = self.M_Distance(X[k,].reshape(1, X.shape[1]), i)
                if self.parameters.loc[i, 'ArousalIndex'] < self.Ta and chistat < self.Tp:
                    if self.parameters.loc[i, 'CompatibilityMeasure'] > compatibility:
                        compatibility = self.parameters.loc[i, 'CompatibilityMeasure']
                        center = i
            if center == -1:
                self.Initialize_Cluster(X[k,], y[k])
                center = self.parameters.last_valid_index()               
            else:
                self.Rule_Update(X[k,], y[k], center)
            for i in self.parameters.index:
                self.Update_Consequent_Parameters(xk, y[k], i)
            if self.parameters.shape[0] > 1:
                self.Merging_Rules(X[k,], center)
            self.rules.append(self.parameters.shape[0])
        return self.OutputTrainingPhase, self.rules
            
    def predict(self, X):
        for k in range(X.shape[0]):
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
            # Compute the compatibility measure and the arousal index for all rules
            Output = 0
            sumCompatibility = 0
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i)
                # Local output
                self.parameters.at[i, 'LocalOutput'] = xk @ self.parameters.loc[i, 'Gamma']
                Output = Output + self.parameters.at[i, 'LocalOutput'] * self.parameters.loc[i, 'CompatibilityMeasure']
                sumCompatibility = sumCompatibility + self.parameters.loc[i, 'CompatibilityMeasure']
            # Global output
            if sumCompatibility == 0:
                Output = 0
            else:
                Output = Output/sumCompatibility
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase
        
    def Initialize_First_Cluster(self, x, y):
        x = x.reshape(1, x.shape[0])
        Q = self.hyperparameters.loc[0, 'omega'] * np.eye(x.shape[1] + 1)
        Gamma = np.insert(np.zeros((x.shape[1],1)), 0, y, axis=0)
        self.parameters = pd.DataFrame([[x, 0., 1., 1., self.Sigma_init, np.array([]), Gamma, Q, 0.]], columns = ['Center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Gamma', 'Q', 'LocalOutput'])
    
    def Initialize_Cluster(self, x, y):
        x = x.reshape(1, x.shape[0])
        Q = self.hyperparameters.loc[0, 'omega'] * np.eye(x.shape[1] + 1)
        Gamma = np.insert(np.zeros((x.shape[1],1)), 0, y, axis=0)
        NewRow = pd.DataFrame([[x, 0., 1., 1., self.Sigma_init, np.array([]), Gamma, Q, 0.]], columns = ['Center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Gamma', 'Q', 'LocalOutput'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
    
    def M_Distance(self, x, i):
        dist = mahalanobis_distance.mahalanobis(x, self.parameters.loc[i, 'Center'], np.linalg.inv(self.parameters.loc[i, 'Sigma']))
        return dist
       
    def Compatibility_Measure(self, x, i):
        x = x.reshape(1, x.shape[0])
        dist = self.M_Distance(x, i)
        self.parameters.at[i, 'CompatibilityMeasure'] = math.exp(-0.5 * dist)
            
    def Arousal_Index(self, x, i):
        x = x.reshape(1, x.shape[0])
        chistat = self.M_Distance(x, i)
        self.parameters.at[i, 'o'] = np.append(self.parameters.loc[i, 'o'], 1) if chistat < self.Tp else np.append(self.parameters.loc[i, 'o'], 0)
        self.parameters.at[i, 'ArousalIndex'] = binom.cdf(sum(self.parameters.loc[i,'o'][-self.hyperparameters.loc[0, 'w']:]), self.hyperparameters.loc[0, 'w'], self.hyperparameters.loc[0, 'lambda1']) if self.parameters.loc[i,'NumObservations'] > self.hyperparameters.loc[0, 'w'] else 0.
    
    def Rule_Update(self, x, y, MaxIndexCompatibility):
        # Update the number of observations in the rule
        self.parameters.loc[MaxIndexCompatibility, 'NumObservations'] = self.parameters.loc[MaxIndexCompatibility, 'NumObservations'] + 1
        # Store the old cluster center
        OldCenter = self.parameters.loc[MaxIndexCompatibility, 'Center']
        G = (self.hyperparameters.loc[0, 'alpha'] * (self.parameters.loc[MaxIndexCompatibility, 'CompatibilityMeasure'])**(1 - self.parameters.loc[MaxIndexCompatibility, 'ArousalIndex']))
        # Update the cluster center
        self.parameters.at[MaxIndexCompatibility, 'Center'] = self.parameters.loc[MaxIndexCompatibility, 'Center'] + G * (x - self.parameters.loc[MaxIndexCompatibility, 'Center'])
        # Updating the dispersion matrix
        self.parameters.at[MaxIndexCompatibility, 'Sigma'] = (1 - G) * (self.parameters.loc[MaxIndexCompatibility, 'Sigma'] - G * (x - self.parameters.loc[MaxIndexCompatibility, 'Center']) @ (x - self.parameters.loc[MaxIndexCompatibility, 'Center']).T)
        
    def Membership_Function(self, x, i):
        dist = distance.mahalanobis(x, self.parameters.loc[i, 'Center'], np.linalg.inv(self.parameters.loc[i, 'Sigma']))
        return math.sqrt(dist)
        
    def Update_Consequent_Parameters(self, xk, y, i):
        self.parameters.at[i, 'Q'] = self.parameters.loc[i, 'Q'] - ((self.parameters.loc[i, 'CompatibilityMeasure'] * self.parameters.loc[i, 'Q'] @ xk.T @ xk @ self.parameters.loc[i, 'Q']) / (1 + self.parameters.loc[i, 'CompatibilityMeasure'] * xk @ self.parameters.loc[i, 'Q'] @ xk.T))
        self.parameters.at[i, 'Gamma'] = self.parameters.loc[i, 'Gamma'] + self.parameters.loc[i, 'Q'] @ xk.T * self.parameters.loc[i, 'CompatibilityMeasure'] * (y - xk @ self.parameters.loc[i, 'Gamma'])
                        
    def Merging_Rules(self, x, MaxIndexCompatibility):
        for i in self.parameters.index:
            if MaxIndexCompatibility != i:
                dist1 = self.M_Distance(self.parameters.loc[MaxIndexCompatibility, 'Center'], i)
                dist2 = self.M_Distance(self.parameters.loc[i, 'Center'], MaxIndexCompatibility)
                if dist1 < self.Tp or dist2 < self.Tp:
                    self.parameters.at[MaxIndexCompatibility, 'Center'] = np.mean(np.array([self.parameters.loc[i, 'Center'], self.parameters.loc[MaxIndexCompatibility, 'Center']]), axis=0)
                    self.parameters.at[MaxIndexCompatibility, 'Sigma'] = [self.Sigma_init]
                    self.parameters.at[MaxIndexCompatibility, 'Q'] = self.hyperparameters.loc[0, 'omega'] * np.eye(x.shape[0] + 1)
                    self.parameters.at[MaxIndexCompatibility, 'Gamma'] = (self.parameters.loc[MaxIndexCompatibility, 'Gamma'] * self.parameters.loc[MaxIndexCompatibility, 'CompatibilityMeasure'] + self.parameters.loc[i, 'Gamma'] * self.parameters.loc[i, 'CompatibilityMeasure']) / (self.parameters.loc[MaxIndexCompatibility, 'CompatibilityMeasure'] + self.parameters.loc[i, 'CompatibilityMeasure'])
                    self.parameters = self.parameters.drop(i)
                    # Stoping to creating new rules when the model exclude the first rule
                    self.ExcludedRule = 1
                