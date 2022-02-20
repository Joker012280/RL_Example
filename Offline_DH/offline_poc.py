import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import plotting
import Buffer

## MountainCar-v0 환경을 이용, 폴이 안쓰러지게 학습
env = gym.make('Cartpole-v0').unwrapped


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, action_size)

    ## 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됨.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 300
num_episodes = 500
input_size = env.observation_space.shape[0]
hidden_size = 64


## 메모리 크기
memory = Buffer.ReplayMemory(200000)

## gym 행동 공간에서 행동의 숫자를 얻기.
n_actions = env.action_space.n

## 네트워크를 초기화한다.
policy_net = DQN(input_size,hidden_size, n_actions)
offline_target_net = DQN(input_size, hidden_size, n_actions)
target_net = DQN(input_size,hidden_size, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

## 최적화 기준 RMS(Root Mean Squeare)
optimizer = optim.RMSprop(policy_net.parameters(), lr =0.001)

## 몇번의 스탭을 했는지.
steps_done = 0

## e- greedy를 이용한 Action 선택
def select_action(state):
    global steps_done
    ## 0과1사이의 값을 무작위로 가져옴
    sample = random.random()
    ## Epsilon-Threshold의 값이 step이 반복될 수록 작아진다.
    ## 점점 greedy action을 택하게 만듬.
    steps_done += 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * (steps_done) / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            """
            t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            기대 보상이 더 큰 행동을 선택할 수 있습니다.
            """
            action = policy_net(Variable(state))

            return torch.FloatTensor(action).data.max(1)[1].view(1,1)
        # Random action 선택
    else:
        return torch.LongTensor([[random.randrange(n_actions)]])

episode_durations = []


## Optimize
def optimize_model_DQN(network,loss_print = False):
    if len(memory) < BATCH_SIZE:
        return
    ## Memory에서 Sample을 가져옴
    transitions = memory.sample(BATCH_SIZE)

    ## Batch로 나눔.
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)

    ## Sample에서 나온 값들을 각각 모음.
    state_batch = torch.cat(state_batch)
    next_state_batch = Variable(torch.cat(next_state_batch))
    action_batch = Variable(torch.cat(action_batch))
    reward_batch = Variable(torch.cat(reward_batch))
    done_batch = Variable(torch.cat(done_batch))

    ## Pseudo Code 12,13 번째 줄 참고!
    state_action_values = network(state_batch).gather(1, action_batch)
    next_state_action_values = network(next_state_batch).max(1)[0]

    ## 기대 Q 값 계산
    expected_state_action_values = reward_batch + (GAMMA * next_state_action_values) * (1 - done_batch)

    ## Huber 손실 계산 / L1 loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    if loss_print :
        print("Loss : ",loss.item())
    ## 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    ## Gradient의 변화가 너무 크지 않게 만듬.
    for param in network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


## 폴이 쓰러지면 한번에 에피소드가 종료
## 이 부분을 여러번 돌리면 학습을 여러번 한 효과

for i_episode in range(num_episodes):
    ## 환경과 상태 초기화
    state = env.reset()
    ## 에피소드 시작 / 폴이 넘어지면 에피소드가 끝남
    for t in count():

        ## 행동 선택과 수행
        state = torch.FloatTensor(np.array([state]))

        action = select_action(state)

        next_state, reward, done, _ = env.step(action.item())

        if done:
            reward = -100

        ## 메모리 저장(한번에 에피소드에 많은 메모리가 생김)
        memory.push((state, action, torch.FloatTensor(np.array([next_state])), \
                     torch.FloatTensor(np.array([reward])), torch.FloatTensor(np.array([done]))))

        if done or t > 500:
            episode_durations.append(t + 1)
            break

            ## 다음 상태로 이동
        state = next_state

        ## 최적화 한단계 수행(Policy 네트워크에서)
        optimize_model_DQN(policy_net)

    ## Output을 계속 볼수 있게
    # if i_episode % 100 == 0:
    #     plotting.plot_durations(episode_durations)
    if i_episode % 1000 == 0:
        clear_output()

plotting.plot_durations(episode_durations,"Training.png")
print('Train Complete')



## Target Network의 복사. 테스트용
target_net.load_state_dict(policy_net.state_dict())

results = []
num_episodes =100
i_episode = 0

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    state = env.reset()

    for t in count():
        # 행동 선택과 수행 / Test하는 과정이므로 학습된 네트워크가 행동 결정
        state = torch.FloatTensor(np.array([state]))

        action = target_net(Variable(state))

        action = torch.FloatTensor(action).data.max(1)[1].view(1,1)

        next_state, reward, done, _ = env.step(action.item())


        if done or t > 500:
            results.append(t + 1)
            break


        # 다음 상태로 이동
        state = next_state
plotting.plot_durations(results,"Online_Result.png")
print('Online Test Complete')

clear_output()

## Testing Offline RL

num_optimize = 5000000
i_optimize = 0
test_num = 0

results = []

for i_optimize in range(num_optimize):
    optimize_model_DQN(offline_target_net)

    if i_optimize % 250000 ==0:
        clear_output()
        print("Num optimize : ",i_optimize)
        print("Testing Start")

        test_num += 1
        for i_episode in range(num_episodes-50) :
            state = env.reset()

            for t in count():
                # 행동 선택과 수행 / Test하는 과정이므로 학습된 네트워크가 행동 결정
                state = torch.FloatTensor(np.array([state]))

                action = offline_target_net(Variable(state))

                action = torch.FloatTensor(action).data.max(1)[1].view(1, 1)

                next_state, reward, done, _ = env.step(action.item())

                if done or t > 500:
                    results.append(t + 1)
                    break

                # 다음 상태로 이동
                state = next_state

        plotting.plot_durations(results,"Offline_Result_" + str(test_num) + ".png")

