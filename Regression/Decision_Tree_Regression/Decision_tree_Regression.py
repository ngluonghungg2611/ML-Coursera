import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
# print(x)
# print(y)

# X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# sc_x = StandardScaler()
# X_train = sc_x.fit_transform(X_train)
# X_test = sc_x.fit(X_test)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

print(y_pred)
# cm = confusion_matrix(y, y_pred)
# print(cm)

plt.scatter(x, y, c='red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Trust of Bluff Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''
    Nhận thấy 2 biến độc lập 
    Entropy: nó tách các biến độc lập lập thành 1 số các khoảng
    Hạn chế: 
        + Trong các khoảng này bài toán Hồi quy cây quyết định không cho được dự đoán mà nó chỉ nối các điểm dự đoán
của 10 mức lương tương ứng với 10 cấp độ
        + Mô hình này không liên tục và đây là mà bài toán không liên tục đầu tiên
'''
  # Để thấy rõ sự không liên tục thì Visualising data
X_grid = np.arange(start=min(x), stop=max(x), step=0.1)
X_gird = X_grid.reshape((len(X_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth of Bluff Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''
    Tương đương với Decision Tree Classification
'''




