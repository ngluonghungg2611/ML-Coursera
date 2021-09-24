import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('Social_Network_Ads.csv.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
# print(dataset)
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)
print(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Pedicting the test set result
y_pred = classifier.predict(X_test)
print(y_pred)