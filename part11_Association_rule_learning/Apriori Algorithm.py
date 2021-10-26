import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
    - Mục đích của bài toán: Tối ưu hóa doanh số bán hàng trong của hàng tạp hóa
    - Sử dụng Association rule learning để biết chính xác vị trí đặt sản phẩm trong của hàng
'''
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# print(dataset)
transactions = []
for i in range(0,7501):
    transactions.append( [ str(dataset.values[i,j]) for j in range(0,20) ] )

print(transactions)









''