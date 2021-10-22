import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
'''
RANDOM FOREST REGRESSION
    - Gồm nhiều Decision Tree Regression tương ứng
    - Các bước thực hiện:
        + Chọn K data points (điểm dữ liệu) ngẫu nhiên trong Training set. Xây dựng 1 cây hồi quy và và sử dụng đưa ra 
        giá trị dự báo sẽ được chỉ định hoăc giá trị y cho bất kì phần tử mới nào  
        + Sau đó xây dựng 1 Decision Tree được liên kết với K data point -> 
        + Quay lại bước 1&2 -> Xây dựng nhiều Decision Tree
        + Đối với mỗi điểm dữ liệu mới, đặt mỗi Tree trong số Ntrees, dự đoán giá trị của Y cho điểm dự liệu được đề cập 
        và chỉ định điểm dữ liệu mới là giá trị trung bình trên tất cả các giá trị Y được dự đoán
        Ví dụ có 500 Tree và nhận được 500 giá trị predict -> Giá trị predict cuối cùng là trung bình cộng của 500 giá trị
        dự đoán
            -> Điều này cải thiện được độ chính xác vì đang lấy giá trị trung bình của nhiều dự đoán
'''
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)
pred = regressor.predict([[6.5]])
print(pred)
# plt.scatter(X,y, color = 'red')
# plt.plot(X, regressor.predict(X), color = 'blue')
# plt.title('Random Forest REgression')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

X_grid = np.arange(start=min(X), stop=max(X), step = 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'r')
plt.plot(X_grid, regressor.predict(X_grid), color='b')
plt.title('RAndom Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''
    Được chia thành nhiều bước hơn vì có nhiều Decision Tree tính toán trong các khoảng
'''