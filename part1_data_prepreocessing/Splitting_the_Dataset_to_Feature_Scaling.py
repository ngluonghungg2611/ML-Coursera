import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('categorical_data.csv')
print('\nBefore process missing data: \n', dataset)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

SimpleImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
SimpleImputer = SimpleImputer.fit(X[:,1:3])
X[:, 1: 3] = SimpleImputer.fit_transform(X[:, 1 : 3])
print('\nAfter process missing data: \n', X)

labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
print('\nAfter label encoder name: \n',X[:,0])

ColumnTransformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
X = ColumnTransformer.fit_transform(X)
print('\n After onehot encoder column name: \n',X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print('\nAfter label encoder column label',y)

#Splitting dataset into the training  set and test set
print('\nAFter splititing dataset: \n')
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print('X_train: \n',X_train)
print('X_test: \n',X_test)
print('y_train: \n',y_train)
print('y_test: \n',y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train  = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X_train)
print(X_test)





