# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:09:18 2018

@author: Eleam Emmanuel
"""

#Upper confidence bound
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
import math

# importing the dataset
dataset = pd.read_csv('Ads.csv')

# Implementing Upper confidence bound
N = 10000
d = 10
ads_selected = []
# initialising the number of selection vector
numbers_of_selections = [0] * d
# initialising the sums of rewards
sums_of_rewards = [0] * d
# lopping through each round
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    # lopping through each instance in a bound
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            # computing the average reward
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            # computing for delta since Upper confidence bound is the sum of delta
            # and average reward
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            # now we compute the upper confidence bound
            upper_bound  = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# visualising the results
plt.hist(ads_selected)
plt.title("Histogram of ads selected")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad wasselected')
plt.show()
