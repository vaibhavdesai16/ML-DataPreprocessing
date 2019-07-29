# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Directory_Of_Homeless_Population_By_Year.csv")
X = dataset.iloc[:, 1:3].values
Y = dataset.iloc[:, 0].values

#adding missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[: , 1:2])
X[:, 1:3] = imputer.transform(X[: , 1:2])


#encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()