import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
print(dataset)
N = 10000
d = 10
ads_selected = []
'''
Bước 1: Ở mỗi vòng n, Xem xét 2 số cho mỗi ad i
    - Ni(n) - số lần ad i được chọn từ i cho đến  n
    - Ri(n) - tổng số reward của ad từ i cho đến  n
'''
numbers_of_selection = [0] * d
sum_of_reward = [0] * d


'''
Bước 2: Từ 2 con số này, chúng ta tính toán:
    - trung bình reward của ad từ i đến n
    - Tính khoảng tin cậy
    -> Cần tính toán 2 con số này ở mỗi vòng
'''
for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if numbers_of_selection[i] > 0: #   lựa chọn khi numbers_of_selection > 0 ít nhất 1 lần
            average_reward = sum_of_reward[i] / numbers_of_selection[i]
            delta_i = math.sqrt(1.5 * math.log(n+1) / numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound =  1e400
        if max_upper_bound < upper_bound :
            max_upper_bound = upper_bound
            ad = i
'''
Bước 3: Chọn ad có độ tin cậy cao nhất
'''