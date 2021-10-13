import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement random selection
N = 10000 # 10000 khach hang
d = 10 # 10 quang cao
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
print(total_reward)
print(ads_selected) # gias tri ad la +1 don vi, vi bat dau tu 0, trong dataset la bat dau tu Ad1
#   Visualising the result - HIstogram
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of time each ad was selected')
plt.show()