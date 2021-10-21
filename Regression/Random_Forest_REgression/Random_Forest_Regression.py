import numpy
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