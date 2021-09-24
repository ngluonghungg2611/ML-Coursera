from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Feature_Scaling.csv')
X = dataset.iloc[:,:-1].values

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)