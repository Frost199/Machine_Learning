# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:49:06 2018

@author: Eleam Emmanuel
"""

# Apriori 
# Importing the libraries

import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

# Importing the dataset with no titles...add headers=None
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transaction = []
for i in range(0, 7501):
    # making a list of transaction from our dataset
    transaction.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
rules = apriori()
