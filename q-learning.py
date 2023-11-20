# This file creates the q_table for Q-learning

import gym
from gym_examples.envs import FieldEnv
from gym.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
from plotcurve import plot_learning_curve
import time

FILENAME = 'qtable.png'

LEARNING_RATE = 0.01
DISCOUNT = 0.99
EPISODES = 10000

SHOW_EVERY = 1000

env = gym.make('gym_examples/GridWorld-v3', )
env = FlattenObservation(env)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high + 1) # divide to 20 chunks or 20 buckets
discrete_os_win_size = (env.observation_space.high - env.observation_space.low + 1) / DISCRETE_OS_SIZE

epsilon = 1.0
EPSILON_DECAY = 1E-6
EPS_END = 0.01
# START_EPSILON_DECAYING = 1
# END_EPSILON_DECAYING = EPISODES // 2
# epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

scores, eps_history = [], []
start_time = time.time()
for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print('Episodes:', episode)
        render = True
    else:
        render = False
    state, info = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    truncated = False
    while not done and not truncated:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done and not truncated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)

            q_table[discrete_state+(action,)] = new_q # update q table

        elif done:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

        # if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        #     epsilon -= epsilon_decay_value
        epsilon = epsilon - EPSILON_DECAY if epsilon > EPS_END \
            else EPS_END
    # print(epsilon)
    scores.append(episode_reward)
    eps_history.append(epsilon)
    ep_rewards.append(episode_reward)

    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
    
    if episode % SHOW_EVERY == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
    
x = [i+1 for i in range(EPISODES)]
plot_learning_curve(x, scores, eps_history, FILENAME)
print("--- %s seconds ---" % (time.time() - start_time))   



