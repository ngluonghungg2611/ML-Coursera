import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# Taking care of missing data
dataset = pd.read_csv('categorical_data.csv')
X = dataset.iloc[:,1:3].values
# print(X)
SimpleImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
SimpleImputer  = SimpleImputer.fit(X)
X = SimpleImputer.transform(X)
# print(X)
dataset.iloc[:,1:3] = X
# print('After process missing data: \n', dataset)

#   Encoding categorical data

Y = dataset.iloc[:,:-1].values
# print(Y)

# print('\n Get label on Name with LableEncoder: \n')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_Y = LabelEncoder()
Y[:,0] = label_encoder_Y.fit_transform(Y[:,0])
# print(Y[:,0])
#
# print('\n Get label on Name with ColumnTransformer: \n')
CT = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [2])], remainder='passthrough')
Y = CT.fit_transform(Y)
print(Y[:,:3])

# onehot encoder label
# print('\n Get label  on label with LabelEncoder: \n')
label = dataset.iloc[:,3].values
# print(label)
label_encoder = LabelEncoder()
label = label_encoder.fit_transform(label)
# print(label)
# print(label_encoder_Y)
# print(22)