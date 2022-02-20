import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

train_data = pd.read_csv("Dataset2.csv")
print(train_data.head())
train_data = train_data.drop(['Date'],axis=1)
robustScaler = MinMaxScaler()
print(robustScaler.fit(train_data))
train_data_robustScaled = robustScaler.transform(train_data)
print(train_data_robustScaled)