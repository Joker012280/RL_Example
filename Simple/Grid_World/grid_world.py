import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


desc=["SFFF", "FFFF", "FFFF", "FFFG"]
## Environment
env = gym.make('FrozenLake-v1',desc = desc,map_name="4x4",is_slippery=False)
state_dim = env.observation_space.n
action_dim = env.action_space.n

Q = np.zeros([state_dim,action_dim])


learning_rate = 0.4
discount_factor = 0.95
num_episodes = 5000

list_total_reward = []

for i in range(num_episodes) :
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    while step < 100:
        step += 1
        action = np.argmax(Q[state, :] + np.random.randn(1, action_dim) * (1. / (i + 1)))

        next_state,reward,done,_ = env.step(action)
        reward = reward -1
        Q[state,action] = Q[state,action] + learning_rate*(reward + discount_factor \
                                                           * np.max(Q[next_state,:]) - Q[state,action])
        total_reward += reward
        state = next_state
        if done :
            break
    list_total_reward.append(total_reward)

print("Average Score : ", (sum(list_total_reward)/num_episodes))

print("Q - Table")
for i in range(action_dim) :
    print("For action : ",i)
    print(np.round(Q[:,i],4).reshape(4,4))

x = np.arange(-0.5,3,1)
y = np.round(Q[:,0],4).reshape(4,4)
plt.pcolormesh(x,x,y)
plt.savefig("Testing.png")