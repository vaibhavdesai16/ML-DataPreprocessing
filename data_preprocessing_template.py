# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:59:20 2019

@author: vaibhav
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv("Directory_Of_Homeless_Population_By_Year.csv")
X = dataset.iloc[:, 1:3].values
Y = dataset.iloc[:, 0].values


#splitting data into the trainning a set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)"""