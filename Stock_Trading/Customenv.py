import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import gym
import yfinance as yf
from collections import deque
from sklearn.preprocessing import MinMaxScaler



class stock_env(gym.Env) :
    metadata = {'render.modes' : ['human']}

    def __init__(self,data,skip,init_money):
        super(stock_env,self).__init__()
        ## For Data
        self.data = data
        self.close = self.data['Close']
        self.obs_data = self.data.drop(['Date','Close'],axis =1)
        ## Data scaling
        self.minmax_scaler = MinMaxScaler()
        self.minmax_scaler.fit(self.obs_data)
        self.obs_data = self.minmax_scaler.transform(self.obs_data)

        self.skip = skip
        self.pos = 0
        self.init_money = init_money
        self.cur_money = init_money
        self.inventory = []
        ## For RL
        self.observation_space = 23
        self.action_space = 3
        self.reward = 0
        self.obs = []
        self.done = False
        ## For masking and else
        self.mask = [1,1,1]
        self.info = []

    def reset(self):
        self.inventory = []
        self.mask = [1,0,1]
        self.pos = 0
        self.obs = []

        self.obs.append(self.obs_data[self.pos])
        self.reward = 0
        self.done = False

        self.cur_money = self.init_money
        self.info = [self.mask]
        return self.obs,self.reward,self.done,self.info

    def get_state(self):
        raise NotImplementedError()

    def get_mask(self):
        if self.cur_money < self.close[self.pos] :
            self.mask[0] = 0
        elif len(self.inventory) == 0 :
            self.mask[1] = 0
        else :
            self.mask = [1,1,1]

    def calculate_reward(self):
        ## TODO : Tune the reward
        sum = len(self.inventory) * self.close[self.pos]
        # print("Calculating Reward :  ", self.pos, sum)
        return 10 * (sum + self.cur_money - self.init_money) / self.init_money

    def step(self,action):

        self.acting(action)
        self.pos += 1
        self.reward = self.calculate_reward()

        self.obs = []
        self.obs.append(self.obs_data[self.pos])

        if self.pos == (len(self.data)-1) :
            self.done = True

        ## TODO : Change Info Status (ex : Invest???)
        self.get_mask()
        self.info = [self.mask]

        return self.obs,self.reward,self.done,self.info

    def acting(self,action):

        # TODO : Purchasing Multiple Stocks

        # Buying
        if action == 0 :
            self.inventory.append(self.close[self.pos])
            self.cur_money -= self.close[self.pos]
            # print("Buying Stock")
        # Selling
        elif action == 1 :
            if len(self.inventory) == 0 :
                raise EnvironmentError
            bought_price = self.inventory.pop(0)
            self.cur_money += self.close[self.pos]
            # print("Selling Stock")
        # DO Nothing
        else :
            pass
            # print("Staying Inventory")

if __name__ == "__main__" :
    ## Get Dataset of Kospi
    # yf.pdr_override()
    # df_full = pdr.get_data_yahoo("^KS11", start="2018-01-01").reset_index()
    # df_full.to_csv('KS11.csv', index=False)
    # ## Pandas Read Dataset
    # df2 = pd.read_csv("Investor2.csv")
    # # for i in df2.columns :
    # #     df2[i] = df2[i].astype('float')
    # df3 = pd.read_csv("KS11.csv")
    # #
    # # ## Modify Data before concatenate
    # df3['Date'] = pd.to_datetime(df3['Date'],format='%Y-%m-%d')
    # df2 = df2.sort_index(ascending=False)
    # df2 = df2.reset_index(drop = True)
    # df3 = df3.drop(['Date'],axis=1)
    # dataset = pd.concat([df2,df3],axis=1)
    # print("----DownLoad Dataset----")
    # dataset.to_csv("Dataset2.csv",index = False)
    # print(dataset.head())
    # print(dataset.columns)
    # print(len(dataset.columns))

    ## Testing Environment
    ## Close means closing price of Kospi
    dataset = pd.read_csv("Dataset2.csv")
    print(dataset.head())
    env = stock_env(dataset,0,10000)
    state,_,_,_ = env.reset()
    print(state)
    for i in range(15) :
        if i < 3 :
            action = 0
        else :
            action = random.randint(0,3)
        print("Action : ",action)
        state,reward,done,info = env.step(action)
        print("State : ", state, " Reward : ",reward, " Done : ",done, " INFO : ",info)


