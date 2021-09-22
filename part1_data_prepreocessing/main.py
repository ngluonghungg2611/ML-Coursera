'''
Missing data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('Book1.csv')
X = dataset.iloc[:,:].values
X = X[:,1:3]
print(X)
simpleIMputer = SimpleImputer(missing_values=np.nan, strategy='mean')
simpleIMputer = simpleIMputer.fit(X)
X = simpleIMputer.transform(X)
print(X)