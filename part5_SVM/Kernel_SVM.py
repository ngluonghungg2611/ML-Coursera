import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import  ListedColormap
from sklearn.svm import SVC

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[0,1]].values
y = dataset.iloc[:,-1].values
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm) # [[64,4],[3,29]] -> 64+29 lần đoán đúng và 3+4 lần đoán sai


X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))

plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X1.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X1.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('RDF kernel (Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

X_set, y_set  = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set==j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
